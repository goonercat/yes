import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

with open("pr3/names.txt", encoding="utf-8", errors="ignore") as f:
    words = [
        parts[0].split()[0].strip()
        for line in f
        if line.strip()
        for parts in [line.split(",")]
        if parts and parts[0].strip()
    ]

words = [words.lower().strip() for words in words if words.strip()]
words = [words for words in words if words.isalpha() and words.isascii()]
words = list(set(words))

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

def bds(words):
    bs = 4
    X, Y = [], []
    for w in words:
        
        context = [0] * bs
        for ch in w + '.':
            ix = stoi [ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape,Y.shape)
    return X,Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,Ytr = bds(words[:n1])
Xdev,Ydev = bds(words[n1:n2])
Xte,Yte = bds(words[n2:])

g = torch.Generator().manual_seed(2147483647)
CharVect = torch.randn((27,15),generator = g)
w1 = torch.randn((60,300),generator = g)
b1 = torch.randn((300),generator = g)
w2 = torch.randn((300,27),generator = g)
b2 = torch.randn((27),generator = g)
paras = [CharVect,w1,b1,w2,b2]


print(len(words))       # how many names?
print(words[:5])        # do they look right?
print(Xtr.shape, Ytr.shape) # is X/Y being built?
print(Xtr.shape, Ytr.shape) # is the split working?

for p in paras:
    p.requires_grad = True



for i in range(20000):# for how many times to train batches of 16

    #Create batches of 16 with replacement (same example can come more than one time)
    ix = torch.randint(0, Xtr.shape[0], (16,))

    #forward pass
    emb = CharVect[Xtr[ix]] # it will pluck out the Charecterstic vector for the selected batch
    h = torch.tanh(emb.view(-1,60)@w1+b1)#flatern emp matrix 4,15,300 -> 60,300 and clac fopr hidden layer
    logits = h@w2+b2#calc for last layer
    loss = F.cross_entropy(logits,Ytr[ix])# calculate loss function {x.exp()/sum(x.exp)}

    #backward pass
    for p in paras:
        p.grad = None
    loss.backward()

    #gradient descent
    if i < 10000:
        lr = 0.1
    else:
        lr = 0.01
    for p in paras:
        p.data -= lr*p.grad


print(loss.item())
    
#training loss
emb = CharVect[Xtr]
h = torch.tanh(emb.view(-1,60)@w1+b1)
logits = h@w2+b2
loss = F.cross_entropy(logits,Ytr)

#Vallidation loss
emb = CharVect[Xdev]
h = torch.tanh(emb.view(-1,60)@w1+b1)
logits = h@w2+b2
loss = F.cross_entropy(logits,Ydev)

#test loss
emb = CharVect[Xte]
h = torch.tanh(emb.view(-1,60)@w1+b1)
logits = h@w2+b2
loss = F.cross_entropy(logits,Yte)

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * 4 # initialize with all ...
    while True:
      emb = CharVect[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ w1 + b1)
      logits = h @ w2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))
