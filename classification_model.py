import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,embed_dim, num_heads=4, mlp_ratio=4.0, dropout_ratio = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim = embed_dim,num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(int(embed_dim*mlp_ratio), embed_dim),
            nn.Dropout(dropout_ratio))
    
    def forward(self,x):
        residual = x
        x  = self.norm(x)
        attn_out, _ = self.attn(x,x,x)
        x = residual + attn_out
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        x = residual + x
        return x

class VLM_Embed_Classifier(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, num_blocks=4, num_classes=2):
        super().__init__()
        self.block = nn.Sequential(
            [Encoder(dim, num_heads, mlp_ratio) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        x = self.block(x)
        x = self.norm(x)
        cls_token = x[:,0,:]
        logits = self.head(cls_token)
        return logits
