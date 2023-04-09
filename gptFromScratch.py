
import torch 
import torch.nn as nn
from torch.nn import functional as F

#------------------------------HYPERPARAMETERS------------------------------

batch_size = 64 #Number of independent sequences processed in parallel
block_size = 256 #Maximum context length for predictions
max_iters = 5000 
eval_interval = 500
learning_rate = 3e-4 
device = 'mps' if torch.backends.mps.is_available() else 'cpu' # grab M1 MAX GPU if available
eval_iters = 200
n_embed = 384 
n_head = 6
n_layer = 6
dropout = 0.2 # Regularization technique, added to avoid overfiting when scaling. Every pass, 20% are calculated go back to 0

#------------------------------CHECK M1 GPU AVAILABILITY------------------------------

if torch.backends.mps.is_available():
	print('Apple M1 Max ready to go')

#------------------------------READ SOURCE FILE------------------------------

with open('Input/legalCorpus.txt', 'r', encoding='utf-8') as f:
	text = f.read()
	
#Sorted of characters in the dataset, #Vocabulary - possible elements of our sequences
chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Vocabulary:\n-----' + ''.join(chars))
print(f'-----\nVocabulary size:\n-----\n {vocab_size}')

#------------------------------ENCODE & DECODE ------------------------------

#Tokenize the input text (convert the raw text to some integers according to a set of rules)
#Encoding function, takes a string and outputs a list of integers
stoi = {ch: i for i, ch in enumerate(chars)} # Iterate on characters and creates a lookup table from character to integer
encode = lambda s: [stoi[c] for c in s] # Translates every character individually with mapping table above
#Decoder, takes a list of integers and outputs a string
itos = {i:ch for i, ch in enumerate(chars)} # Iterate on characters and creates a lookup table from integer to character
decode = lambda l: ''.join([itos[i] for i in l]) # Translates every integer individually with mapping table above

#------------------------------TRAIN AND TEST SPLITS------------------------------

data = torch.tensor(encode(text), dtype=torch.long)
# Separate train and validation
n = int(0.9*len(data))
train_data = data[:n] # first 90% data
val_data = data[n:] # remaining 10% for validation

#------------------------------DATA LOADER------------------------------

def get_batch(split):
	# Generate a small batch of data of inputs x and targets y
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,)) #batch_size random positions to grab
	x = torch.stack([data[i:i+block_size] for i in ix]) #first block_size charaters starting at i and use torch.stack to take 1-dimensional tensors and stack them up at rows in a 4 x 8 tensor
	y = torch.stack([data[i+1:i+block_size+1] for i in ix])
	x,y = x.to(device), y.to(device) #when we load, move to device (GPU)
	return x,y

#------------------------------ESTIMATE LOSS------------------------------

@torch.no_grad() #context manager, telling pytorch that everything happening in the function we will not call .backward on (back propagation), so python can be more efficient on memory use because we will never call backward. 
def estimate_loss(): #averages the loss on multiple iterations (eval_iters)
	out = {}
	model.eval() #model is set to evaluation phase
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(split)
			logits, loss = model(X,Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train() #model is re-set to train phase
	return out

#------------------------------SELF ATTENTION------------------------------

class Head(nn.Module):
	""" one head of self-attention
	In the constructor, you declare all the layers you want to use.
	In the forward function, you define how your model is going to be run, from input to output
	"""
	
	def __init__(self, head_size): # pass the head size
		super().__init__()
		self.key = nn.Linear(n_embed, head_size, bias=False) #linear layers, typically we don't use biases here
		self.query = nn.Linear(n_embed, head_size, bias=False)
		self.value = nn.Linear(n_embed, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #tril is a buffer, not a parameter. Its assigned using register buffer, that's the tril - the lower piramid vector

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		B, T, C = x.shape 
		k = self.key(x) # (B, T, hs)
		q = self.query(x) # (B, T, hs)
		#compute attention scores ('affinities')
		wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T) # attention scores, which we then normalize using the C**-0.5
		wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #(B, T, T) # we make sure it does communicate with the past
		wei = F.softmax(wei, dim=-1) #(B, T, T)
		wei = self.dropout(wei)
		# aggregate value and output
		v = self.value(x) #(B, T, C) 
		out = wei @ v #(B, T, T) @ #(B, T, C) -> #(B, T, C)
		return out

class MultiHeadAttention(nn.Module):
	"""Multiple heads of self-attention in parallel"""

	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #Creating multiple heads
		self.proj = nn.Linear(head_size * num_heads, n_embed)
		self.dropout = nn.Dropout(dropout)

	def forward(self,x):
		out = torch.cat([h(x) for h in self.heads], dim=-1) #Concatenate all outputs on channel dimension (dim-1)
		out = self.dropout(self.proj(out)) #linear transformation of the outcome above
		return out 

class FeedForward(nn.Module):
	'''A simple linear layer followed by a non linearity'''
	def __init__(self, n_embed):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embed, 4 * n_embed),
			nn.ReLU(),
			nn.Linear(4 * n_embed, n_embed), 
			nn.Dropout(dropout),
		)

	def forward(self, x): #Per token level, once they gathered all the detail they think of each data individually
		return self.net(x)

class Block(nn.Module):
	'''Transformer block: communication followed by computation'''

	def __init__(self, n_embed, n_head) -> None:
		#n_embed: embedding dimensions, n_head, the number of heads we'd like
		super().__init__()
		head_size = n_embed // n_head 
		self.sa = MultiHeadAttention(n_head, head_size)
		self.ffwd = FeedForward(n_embed)
		self.ln1 = nn.LayerNorm(n_embed) #normalizing per token, makes them gausean
		self.ln2 = nn.LayerNorm(n_embed) #normalizing per token, makes them gausean

	def forward(self, x):
		x = x + self.sa(self.ln1(x)) # Residual connections
		x = x + self.ffwd(self.ln2(x))
		return x

#------------------------------BIGRAM MODEL WE DEVELOPED------------------------------

class GPTLanguageModel(nn.Module): #subclass of nn.Module
		
	def __init__(self):
		super().__init__()
		#each token directly reads off the logits for the next token from a lookup table
		self.token_embedding_table = nn.Embedding(vocab_size, n_embed) #n_embed is the number of embedding dimensions
		self.position_embedding_table = nn.Embedding(block_size, n_embed)
		self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embed) # final layer norm
		self.lm_head = nn.Linear(n_embed, vocab_size) #lm_head short for language modeling head
		
		# better init, not covered in the original GPT video, but important, will cover in followup video
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, idx, targets=None):
		B, T = idx.shape
		# idx and targets are both (B,T) tensor of integers
		tok_emb = self.token_embedding_table(idx) # (Batch = 4, Time = 8, Channels (vocab size) = 64) 
		pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) 
		# logits are the scores for the next character of the sequence
		x = tok_emb + pos_emb #(B,T,C)
		x = self.blocks(x) # (B, T, C)
		x = self.ln_f(x) # (B, T, C)
		logits = self.lm_head(x) #(B, T, vocab_size)
		
		if targets is None:
			loss = None
		else:
			#Reshape to be able to execute cross_entropy requirements
			B, T, C = logits.shape
			logits = logits.view(B*T,C)
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets) #Measure loss function quality of logits in respect to targets
		
		return logits, loss
	
	def generate(self, idx, max_new_tokens): #idx is the current context of characters, job is to continue extending it until it reaches max_new_tokens
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			# we can't never have more than block size tokens, so we need to add a crop
			idx_cond = idx[:, -block_size:]
			#get prediction
			logits, loss = self(idx_cond) #self(idx) will end up calling forward, note we're not giving any targets
			#focus only on the last time step
			logits = logits[:, -1, :] # becomes (B, C)
			# apply softmax to get probabilities
			probs = F.softmax(logits, dim=-1) # (B,C)
			#sample from the distribution
			idx_next = torch.multinomial(probs, num_samples=1) # (B,1) Given num_spaces is 1, we'll get 1
			#append sampled index to the running sequence
			idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
		return idx
	
model = GPTLanguageModel()
m = model.to(device) #move parameters to device

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters') 
# logits, loss = m(xb, yb) #Passing inputs and targets
# print(logits.shape)
# print(loss)

#------------------------------OPTIMIZER------------------------------

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate) #typical rating for learning rate is 3e-4, but for small networks you can get away with much higher learning rates like 1e-3

#------------------------------TRAINING LOOP------------------------------

for iter in range(max_iters):
	#every once in a while evaluate the loss in train and val sets
	if iter % eval_interval == 0:
		lossess = estimate_loss()
		print(f"Step {iter}: train loss {lossess['train']:.4}, val loss {lossess['val']:.4f}")
		
	xb, yb = get_batch('train')
	
	#evaluate the loss
	logits, loss = m(xb, yb) #evaluating the loss
	optimizer.zero_grad(set_to_none=True) #Zeroing gradients from previous step
	loss.backward() #getting gradients from all parameters
	optimizer.step() #updating parameters


#------------------------------GENERATE FROM MODEL------------------------------

context = torch.zeros((1,1), dtype=torch.long, device=device) #note added device for GPU

torch.save(m,'outputModel.pt')


print(decode(m.generate(context , max_new_tokens=500)[0].tolist())) 
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
