"""
src/pretrain/model.py
Masked Image Modeling (MIM) architecture.
  Encoder : ViT-Base (12 layers, 768 dim, 12 heads)
  Decoder : lightweight (4 layers, 512 dim, 16 heads)
  Input   : 15-channel patches (256x256 px)
  Patches : 16x16 px tokens
  Masking : 75% random
  Loss    : MSE on masked patches only
"""

import torch
import torch.nn as nn


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    assert embed_dim % 4 == 0
    half  = embed_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(0, half, 2, dtype=torch.float32) / half))

    h_ids = torch.arange(grid_h, dtype=torch.float32)
    w_ids = torch.arange(grid_w, dtype=torch.float32)

    h_sin = torch.sin(h_ids[:, None] * omega[None, :])
    h_cos = torch.cos(h_ids[:, None] * omega[None, :])
    w_sin = torch.sin(w_ids[:, None] * omega[None, :])
    w_cos = torch.cos(w_ids[:, None] * omega[None, :])

    h_emb = torch.cat([h_sin, h_cos], dim=-1)[:, None, :].expand(grid_h, grid_w, half)
    w_emb = torch.cat([w_sin, w_cos], dim=-1)[None, :, :].expand(grid_h, grid_w, half)

    return torch.cat([h_emb, w_emb], dim=-1).reshape(grid_h * grid_w, embed_dim)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=15, embed_dim=768):
        super().__init__()
        self.grid_size   = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj        = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = self.drop((q @ k.transpose(-2,-1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1,2).reshape(B, N, C))


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=15,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.grid_size   = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))

        pos = get_2d_sincos_pos_embed(embed_dim, self.grid_size, self.grid_size)
        self.register_buffer("pos_embed", pos.unsqueeze(0))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.cls_token, std=0.02)

    def random_masking(self, x, mask_ratio=0.75):
        B, N, D = x.shape
        n_keep  = int(N * (1 - mask_ratio))
        noise   = torch.rand(B, N, device=x.device)
        ids_shuffle  = torch.argsort(noise, dim=1)
        ids_restore  = torch.argsort(ids_shuffle, dim=1)
        ids_keep     = ids_shuffle[:, :n_keep]
        x_visible    = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1,-1,D))
        mask         = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_visible, mask, ids_restore

    def forward(self, x, mask_ratio=0.75):
        x = self.patch_embed(x) + self.pos_embed
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, ids_restore


class MIMDecoder(nn.Module):
    def __init__(self, num_patches, encoder_embed_dim=768, decoder_embed_dim=512,
                 decoder_depth=4, decoder_num_heads=16, patch_size=16,
                 in_channels=15, mlp_ratio=4.0, grid_size=16):
        super().__init__()
        patch_dim   = patch_size * patch_size * in_channels
        self.embed  = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        pos = get_2d_sincos_pos_embed(decoder_embed_dim, grid_size, grid_size)
        self.register_buffer("pos_embed", pos.unsqueeze(0))

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.pred = nn.Linear(decoder_embed_dim, patch_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, latent, ids_restore):
        x = self.embed(latent)[:, 1:, :]         # remove CLS
        B, N_vis, D = x.shape
        n_masked = ids_restore.shape[1] - N_vis
        mask_tokens = self.mask_token.expand(B, n_masked, D)
        x_full = torch.gather(
            torch.cat([x, mask_tokens], dim=1), 1,
            ids_restore.unsqueeze(-1).expand(-1,-1,D)
        )
        x_full = x_full + self.pos_embed
        for block in self.blocks:
            x_full = block(x_full)
        return self.pred(self.norm(x_full))


class MaskedImageModel(nn.Module):
    """
    Full MIM model: ViT-Base encoder + lightweight decoder.
    Proposal section 4.2.1.
    """

    def __init__(self, img_size=256, patch_size=16, in_channels=15,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mask_ratio=0.75, mlp_ratio=4.0):
        super().__init__()
        self.mask_ratio  = mask_ratio
        self.patch_size  = patch_size
        self.in_channels = in_channels
        self.grid_size   = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.encoder = ViTEncoder(img_size, patch_size, in_channels,
                                   encoder_embed_dim, encoder_depth,
                                   encoder_num_heads, mlp_ratio)
        self.decoder = MIMDecoder(self.num_patches, encoder_embed_dim,
                                   decoder_embed_dim, decoder_depth,
                                   decoder_num_heads, patch_size, in_channels,
                                   mlp_ratio, self.grid_size)

    def patchify(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        G = H // P
        return x.reshape(B, C, G, P, G, P).permute(0,2,4,3,5,1).reshape(B, G*G, P*P*C)

    def compute_loss(self, pred, target, mask):
        loss = ((pred - target) ** 2).mean(dim=-1)
        return (loss * mask).sum() / (mask.sum() + 1e-8)

    def forward(self, x):
        latent, mask, ids_restore = self.encoder(x, self.mask_ratio)
        pred   = self.decoder(latent, ids_restore)
        target = self.patchify(x)
        loss   = self.compute_loss(pred, target, mask)
        return loss, pred, mask

    def encode(self, x, mask_ratio=0.0):
        latent, _, _ = self.encoder(x, mask_ratio=mask_ratio)
        return latent


def build_model(config):
    return MaskedImageModel(
        img_size           = config.get("img_size", 256),
        patch_size         = config.get("patch_size", 16),
        in_channels        = config.get("in_channels", 15),
        encoder_embed_dim  = config.get("encoder_embed_dim", 768),
        encoder_depth      = config.get("encoder_depth", 12),
        encoder_num_heads  = config.get("encoder_num_heads", 12),
        decoder_embed_dim  = config.get("decoder_embed_dim", 512),
        decoder_depth      = config.get("decoder_depth", 4),
        decoder_num_heads  = config.get("decoder_num_heads", 16),
        mask_ratio         = config.get("mask_ratio", 0.75),
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model  = MaskedImageModel().to(device)
    total  = sum(p.numel() for p in model.parameters())
    enc    = sum(p.numel() for p in model.encoder.parameters())
    dec    = sum(p.numel() for p in model.decoder.parameters())
    print(f"Total params  : {total/1e6:.1f}M")
    print(f"Encoder params: {enc/1e6:.1f}M")
    print(f"Decoder params: {dec/1e6:.1f}M")
    x = torch.randn(2, 15, 256, 256).to(device)
    loss, pred, mask = model(x)
    print(f"Loss: {loss.item():.4f}  Pred: {pred.shape}  Mask: {mask.shape}")
    print("Model OK")
