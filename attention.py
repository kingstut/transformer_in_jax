import jax.numpy as jnp
import flax.linen as nn

class Attention(nn.Module):
    '''
    vanilla multi-head masked self-attention layer with a projection at the end;
    inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py; 
    flax version of torch.nn.MultiheadAttention
    '''

    n_embd: int
    num_head: int = 8 
    attn_pdrop: float = 0
    resid_pdrop: float = 0

    '''
    @nn.compact allows you to define your whole module in a single method, 
    and “co-locate” submodules and variables next to where they are used  
    '''
    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = nn.Dense(3 * self.n_embd)(x).split(C, dim=2)

        #reshape to (B, nh, T, hs)
        k = jnp.reshape(k, (B, T, self.n_head, C // self.n_head)).transpose((1, 2))
        q = jnp.reshape(q, (B, T, self.n_head, C // self.n_head)).transpose((1, 2))
        v = jnp.reshape(v, (B, T, self.n_head, C // self.n_head)).transpose((1, 2))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ jnp.transpose(k, (-2, -1))) * (1.0 / (k.size(-1))**0.5)
        
        #FIXME att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att= nn.softmax(att, axis=-1)
        att = nn.Dropout(self.attn_pdrop)(att)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).reshape(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = nn.Dense(self.n_embd)(y)
        y = nn.Dropout(self.attn_pdrop)(y)
        return y



