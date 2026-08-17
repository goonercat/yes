
import torch
import torch.nn as nn
import math

# target would also require embeding and positional encoding as we are translating from one language to another
#  and hence we need to give the target sentence to the model as well.

# d_model = length of embeding vector for each token

# nn.module is Inheritance

# Embedding matrix

class Input_Embedings(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        # To make embeding vector
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        # embeding * root(d) 
        return self.embedding(x) * math.sqrt(self.d_model) # we return the embeding


# Positional Encoding

# to give model the info about positions of the words in a sentence as all the 
# words are to nn given at the same.
# sin(2i) = pos/(10000Pow(2i/d_model))  and cos of same thing for odd positions
class PositionalEncoding(nn.Module):
    
# Dropout randomly "drops out" (or ignores) a certain percentage of the neurons or connections in a layer.
# By doing this, it forces the model to learn more robust features rather than relying too heavily on any 
# single neuron or specific subset of features


    def __init__(self,d_model:int,seq_len:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Matrix of seq_len * d_model
        pe = torch.zeros(seq_len,d_model)
        # vector of (seq_len * 1) form 0 to seq_len-1
        position = torch.arange(0,seq_len,dtype = torch.float).unsqueeze(1)
        # make (seq_len,1) vector for the denominator
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        # Apply sin to even position
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # increase a dimention in 0th dimention (1,seq_len,d_model)
        # to tell that this is not a parameter 
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x) # dropout applied to the sum of embeding and positional encoding


# Normalization layer

class LayerNormalization(nn.Module):

    def __init__(self, d_model:int, eps:float = 10**-6)->None:
        # epision prevents division by zero if sigma is close to zero
        super().__init__()
        self.eps = eps
        # nn.parameter allows the values to be
        # learned during training 
        self.alpha = nn.Parameter(torch.ones(d_model)) # multiplied
        self.bias = nn.Parameter(torch.zeros(d_model)) # added

    def forward(self,x):
        mean = x.mean(dim = -1,keepdim = True)
        var = x.std(dim = -1,keepdim = True)
        return self.alpha * (x-mean) /(var + self.eps) + self.bias


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)  # root mean square normalization
        return self.weight * (x / (norm + self.eps))

# Neural Network

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) # w1 b1 (biases are automatically made)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model) # w2 b2 (biases are automatically made)
        
    
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) # dropout is applied to the output of the first linear layer after ReLU activation
    

# Multi-Head atention

class MultiHeadAttentionBlock(nn.Module):

# input matrix is seq * d_model nad isnt embedings

    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.h = h
        self.d_model = d_model

        #now we must divide the embedings in h heads so d_model must be divisible by h.
        assert d_model % h == 0, "d_model is not divisible by h" # throws error if not div.
        
        self.d_k = d_model // h

        # define weights
        self.w_q = nn.Linear(d_model,d_model, bias = False)
        self.w_k = nn.Linear(d_model,d_model, bias = False)
        self.w_v = nn.Linear(d_model,d_model, bias = False)

        self.w_o = nn.Linear(d_model,d_model, bias = False)
        self.dropout = nn.Dropout(dropout)

    # Flash Attention
    @staticmethod
    def attention(query, key, value, mask, dropout, is_causal=False):
        dropout_p = dropout.p if dropout is not None else 0.0
        
        x = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
        
        return x, None

    
    # q,k,v are differently used insted of just using x so that we can even use it in 
    # cross atention.
    def forward(self,q,k,v,mask=None, is_causal=False):
        # (batch,seq_len,d_model)
        query = self.w_q(q) # w_q(q) calls w_q.forward(q)and does v @ W_v^T + b
        key = self.w_k(k)
        value = self.w_v(v)
        #                           we divided it in h parts
        # (batch,seq_len,d_model) --> (Batch,seq_len,h,d_k) --> (Batch,h,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value, mask, self.dropout,is_causal=is_causal)

        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)

        return self.w_o(x)


# Residual connection

# (we skip training through a layer and add it directly to output)  
# Normalizes and adds output to initial input

class ResidualConnection(nn.Module):

    def __init__(self, d_model: int,dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class DecoderBlock(nn.Module):
    
    def __init__(self, d_model: int, self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(d_model,dropout) for _ in range(2)])
        # modulelist handels modules, we made a list of residual connections

    def forward(self,x,tgt_mask):
        # is causal = true
        #                                                                            masks the tokens ahead
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, None, is_causal=True))
        # it is self attention and hence all q,k,v are x.
        x = self.residual_connection[1](x,self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(d_model)

    def forward(self, x, tgt_mask):
        # we take n decoder blocks and pass the output of one to the next
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # batch,seq_len,d_model --> batch,seq_len,vocab_size
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.projection(x)


class Transformer(nn.Module):
    def __init__(self, decoder: Decoder, tgt_embed: Input_Embedings, tgt_pos:PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    
    def decode(self, tgt, tgt_mask):
        # we put input in embedding and add positional encoding to it and then pass it to decoder
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, tgt_mask)


    def project(self, x):
        return self.projection_layer(x)

    





def build_transformer( tgt_vocab_size: int, d_model: int = 768, d_ff: int = 4096, h: int = 12, num_decoder_layers: int = 12, dropout: float = 0.1, tgt_seq_len: int = 768) -> Transformer:

    tgt_embed = Input_Embedings(d_model, tgt_vocab_size)

    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)


    
    decoder_blocks = []
    for _ in range(num_decoder_layers):
        # make self attention block, cross attention block and feed forward block and then make decoder block from them and add
        # it to the list of decoder blocks
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(d_model, self_attention_block, feed_forward_block, dropout))
    
    # create encoder and decoder from the list of encoder and decoder blocks
    decoder = Decoder(d_model,nn.ModuleList(decoder_blocks))

    # Create projection layer to map decoder output to target vocabulary size
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the Transformer 
    transformer = Transformer(decoder, tgt_embed, tgt_pos, projection_layer)

    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


# aprox chunk size = 256 & k = 3
#
model = build_transformer(
    tgt_vocab_size=10000,
    d_model=768,
    d_ff=4096,
    h=12,
    num_decoder_layers=12,
    tgt_seq_len=768
)
print("Works! ✅")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
