"""
src/finetune/model_fusion.py
Multi-Modal Fusion Prospectivity Model (proposal section 4.3).
  Geological branch  : pre-trained ViT on channels 0-7
  Geophysical branch : pre-trained ViT on channels 8-11
  Feature branch     : MLP on channels 12-14
  CCA fusion         : deep CCA + 3-layer MLP
  Cross-attention    : geology <-> geophysics
  Depth heads        : 3 classifiers for 0-500m, 500-1000m, 1000-2000m
  MC Dropout         : p=0.2 for uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pretrain.model import ViTEncoder

GEO_CHANNELS   = list(range(0, 8))
AERO_CHANNELS  = list(range(8, 12))
FEAT_CHANNELS  = list(range(12, 15))
DEPTH_INTERVALS = ["0_500m", "500_1000m", "1000_2000m"]


class DeepCCAFusion(nn.Module):
    def __init__(self, in_dim, cca_dim=256, out_dim=512, dropout=0.2):
        super().__init__()
        self.geo_proj = nn.Sequential(
            nn.Linear(in_dim, cca_dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(cca_dim*2, cca_dim), nn.LayerNorm(cca_dim),
        )
        self.aero_proj = nn.Sequential(
            nn.Linear(in_dim, cca_dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(cca_dim*2, cca_dim), nn.LayerNorm(cca_dim),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(cca_dim*2, out_dim), nn.GELU(), nn.LayerNorm(out_dim), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),   nn.GELU(), nn.LayerNorm(out_dim), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),              nn.LayerNorm(out_dim),
        )

    def forward(self, geo, aero):
        gp = self.geo_proj(geo)
        ap = self.aero_proj(aero)
        return self.fusion_mlp(torch.cat([gp, ap], dim=-1)), gp, ap


class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.q_geo = nn.Linear(dim, dim); self.k_aero = nn.Linear(dim, dim)
        self.v_aero = nn.Linear(dim, dim); self.out_geo = nn.Linear(dim, dim)
        self.q_aero = nn.Linear(dim, dim); self.k_geo = nn.Linear(dim, dim)
        self.v_geo = nn.Linear(dim, dim);  self.out_aero = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)

    def _attn(self, q, k, v, B, N):
        H, D = self.num_heads, self.head_dim
        q = q.reshape(B,N,H,D).transpose(1,2); k = k.reshape(B,N,H,D).transpose(1,2)
        v = v.reshape(B,N,H,D).transpose(1,2)
        return (self.drop((q@k.transpose(-2,-1))*self.scale).softmax(dim=-1)@v).transpose(1,2).reshape(B,N,H*D)

    def forward(self, geo, aero):
        B, N, D = geo.shape
        geo_out  = self.norm1(geo  + self.out_geo( self._attn(self.q_geo(geo),   self.k_aero(aero), self.v_aero(aero), B, N)))
        aero_out = self.norm2(aero + self.out_aero(self._attn(self.q_aero(aero), self.k_geo(geo),   self.v_geo(geo),   B, N)))
        return geo_out, aero_out


class FeatureBranch(nn.Module):
    def __init__(self, in_channels=3, out_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim),     nn.GELU(), nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.net(x.mean(dim=(-2,-1)))


class ProspectivityHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),    nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class MineralProspectivityModel(nn.Module):
    GEO_CH = GEO_CHANNELS; AERO_CH = AERO_CHANNELS; FEAT_CH = FEAT_CHANNELS

    def __init__(self, pretrain_ckpt=None, img_size=256, patch_size=16,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 cca_dim=256, fusion_dim=512, feat_dim=128,
                 mc_dropout=0.2, n_depth_levels=3):
        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim

        self.geo_encoder  = ViTEncoder(img_size, patch_size, len(self.GEO_CH),
                                        encoder_embed_dim, encoder_depth, encoder_num_heads)
        self.aero_encoder = ViTEncoder(img_size, patch_size, len(self.AERO_CH),
                                        encoder_embed_dim, encoder_depth, encoder_num_heads)
        self.cross_attn   = CrossModalAttention(encoder_embed_dim, 8, 0.1)
        self.feat_branch  = FeatureBranch(len(self.FEAT_CH), feat_dim, mc_dropout)
        self.cca_fusion   = DeepCCAFusion(encoder_embed_dim, cca_dim, fusion_dim, mc_dropout)
        self.depth_heads  = nn.ModuleList([
            ProspectivityHead(fusion_dim + feat_dim, 256, mc_dropout)
            for _ in range(n_depth_levels)
        ])
        if pretrain_ckpt is not None:
            self._load_pretrained(pretrain_ckpt)

    def _load_pretrained(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"WARNING: {ckpt_path} not found, training from scratch")
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        enc_state = {k.replace("encoder.",""): v
                     for k,v in ckpt["model"].items() if k.startswith("encoder.")}
        for enc, name in [(self.geo_encoder,"geo"), (self.aero_encoder,"aero")]:
            own = enc.state_dict(); n = 0
            for k, v in enc_state.items():
                if k not in own or "patch_embed.proj" in k: continue
                if own[k].shape != v.shape: continue
                own[k].copy_(v); n += 1
            enc.load_state_dict(own)
            print(f"  {name} encoder: {n} weight tensors loaded from pre-trained")

    def freeze_encoders(self):
        for p in list(self.geo_encoder.parameters()) + list(self.aero_encoder.parameters()):
            p.requires_grad = False
        print("  Encoders frozen")

    def unfreeze_encoders(self):
        for p in list(self.geo_encoder.parameters()) + list(self.aero_encoder.parameters()):
            p.requires_grad = True
        print("  Encoders unfrozen")

    def forward(self, x):
        geo_lat,  _, _ = self.geo_encoder( x[:, self.GEO_CH,  :, :], mask_ratio=0.0)
        aero_lat, _, _ = self.aero_encoder(x[:, self.AERO_CH, :, :], mask_ratio=0.0)
        geo_tok,  aero_tok  = self.cross_attn(geo_lat[:,1:,:], aero_lat[:,1:,:])
        geo_cls  = geo_lat[:, 0, :]
        aero_cls = aero_lat[:, 0, :]
        feat_out = self.feat_branch(x[:, self.FEAT_CH, :, :])
        fused, gp, ap = self.cca_fusion(geo_cls, aero_cls)
        combined = torch.cat([fused, feat_out], dim=-1)
        out = {"geo_proj": gp, "aero_proj": ap, "combined": combined}
        for head, depth in zip(self.depth_heads, DEPTH_INTERVALS):
            out[f"logits_{depth}"] = head(combined)
        return out

    def predict_proba(self, x, mc_passes=50):
        self.train()
        results = {d: [] for d in DEPTH_INTERVALS}
        with torch.no_grad():
            for _ in range(mc_passes):
                out = self.forward(x)
                for d in DEPTH_INTERVALS:
                    results[d].append(torch.sigmoid(out[f"logits_{d}"]))
        self.eval()
        return {d: {"mean": torch.stack(results[d],-1).mean(-1),
                    "std":  torch.stack(results[d],-1).std(-1),
                    "all_pass": torch.stack(results[d],-1)} for d in DEPTH_INTERVALS}


def cca_loss(geo_proj, aero_proj, reg=1e-4):
    B, D = geo_proj.shape
    gc = geo_proj  - geo_proj.mean(0, keepdim=True)
    ac = aero_proj - aero_proj.mean(0, keepdim=True)
    Sxy = (gc.T @ ac) / (B-1)
    Sxx = (gc.T @ gc) / (B-1) + reg * torch.eye(D, device=geo_proj.device)
    Syy = (ac.T @ ac) / (B-1) + reg * torch.eye(D, device=geo_proj.device)
    corr = torch.trace(Sxy) / (torch.sqrt(torch.trace(Sxx)) * torch.sqrt(torch.trace(Syy)) + 1e-8)
    return -corr


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MineralProspectivityModel(
        pretrain_ckpt="experiments/pretrain_mim/checkpoint_best.pt"
    ).to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {total:.1f}M")
    x = torch.randn(2, 15, 256, 256).to(device)
    out = model(x)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    print("Model OK")
