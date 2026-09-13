import torch 
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    def __init__(self,img_size,patch_size,in_channels,d_model):
        super().__init__()
        assert img_size%patch_size==0
        self.img_size=img_size
        self.patch_size=patch_size
        self.num_patches=(img_size//patch_size)**2
        self.projection=nn.Conv2d(in_channels=in_channels,out_channels=d_model,kernel_size=patch_size,stride=patch_size)
    def forward(self,x):
        x=self.projection(x)
        x=x.flatten(2)
        x=x.transpose(1,2)
        return x
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model%num_heads==0
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads
        self.qkv=nn.Linear(d_model,3*d_model)
        self.out_proj=nn.Linear(d_model,d_model)
    def forward(self,x):
        B,N,D=x.shape
        qkv=self.qkv(x)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        qkv=qkv.permute(2,0,3,1,4)# (3, B, H, N, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]# q, k, v: (B, H, N, Hd)
        attn=q@k.transpose(-1,-2)/math.sqrt(q.size(-1))
        attn=F.softmax(attn,dim=-1)
        output=attn@v
        out=output.transpose(1,2)
        out=out.contiguous().view(B,N,D)
        out=self.out_proj(out)
        return out

class MLP(nn.Module):
    def __init__(self,d_model,mlp_ratio):
        super().__init__()
        hidden_dim=d_model*mlp_ratio
        self.net=nn.Sequential(nn.Linear(d_model,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,d_model))
    def forward(self,x):
        return self.net(x)
class ViTBlock(nn.Module):
    def __init__(self,d_model,num_heads,mlp_ratio=4):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model)
        self.attn=MultiHeadAttention(d_model,num_heads)
        self.ln2=nn.LayerNorm(d_model)
        self.mlp=MLP(d_model,mlp_ratio)
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        x=x+self.mlp(self.ln2(x))
        return x

class TinyVit(nn.Module):
    def __init__(self,img_size=32,patch_size=8,in_channel=3,d_model=64,num_heads=16,num_layers=2,mlp_ratio=4):
        super().__init__()
        self.patch_embed=PatchEmbedding(img_size,patch_size,in_channel,d_model)
        self.num_patches=self.patch_embed.num_patches
        self.cls_token=nn.Parameter(torch.zeros(1,1,d_model))
        self.pos_embed=nn.Parameter(torch.zeros(1,self.num_patches+1,d_model))
        self.blocks=nn.ModuleList([ViTBlock(d_model,num_heads,mlp_ratio) for _ in range(num_layers)])
        self.ln=nn.LayerNorm(d_model)
    def forward(self,img):
        x=self.patch_embed(img)
        B,N,D=x.shape
        cls_token=self.cls_token.expand(B,-1,-1)
        x=torch.cat([cls_token,x],dim=1)
        x=x+self.pos_embed[:,:N+1,:]
        for block in self.blocks:
            x=block(x)
        x=self.ln(x)
        return x

