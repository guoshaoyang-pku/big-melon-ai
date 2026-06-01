"""Policy / value networks (PyTorch, MPS-ready).

The public training sample format remains the historical flat vector so old
replay files stay usable. ``TransformerPolicyValueNet`` converts that vector
into a global token plus masked fruit tokens inside ``forward``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import FRUIT_FEATS, NUM_FRUIT_TYPES, global_feat_count, input_dim


class PolicyValueNet(nn.Module):
    """Legacy flattened MLP kept for old checkpoints and comparisons."""

    def __init__(self, in_dim, K, hidden=384, n_layers=3):
        super().__init__()
        self.in_dim = in_dim
        self.K = K
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden, K)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def infer_batch(self, vecs, device):
        self.eval()
        t = torch.as_tensor(vecs, dtype=torch.float32, device=device)
        logits, values = self.forward(t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs, values.cpu().numpy()

    @torch.no_grad()
    def infer(self, vec, device):
        self.eval()
        t = torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(t)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, float(value.item())


class TransformerPolicyValueNet(nn.Module):
    """Object Transformer over one global token plus masked fruit tokens."""

    def __init__(self, K, max_fruits, d_model=128, n_heads=4,
                 n_layers=2, dropout=0.1, ff_mult=4,
                 boundary_features=False):
        super().__init__()
        self.K = int(K)
        self.max_fruits = int(max_fruits)
        self.boundary_features = bool(boundary_features)
        self.global_feats = global_feat_count(self.boundary_features)
        self.in_dim = input_dim(self.max_fruits, self.boundary_features)
        self.d_model = int(d_model)
        self.global_proj = nn.Linear(self.global_feats, self.d_model)
        self.fruit_proj = nn.Linear(FRUIT_FEATS, self.d_model)
        self.type_embed = nn.Embedding(NUM_FRUIT_TYPES, self.d_model)
        self.cls_bias = nn.Parameter(torch.zeros(1, 1, self.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(self.d_model * ff_mult),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=int(n_layers), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.d_model)
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.K),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    def _split_flat(self, x):
        if x.ndim != 2 or x.shape[-1] != self.in_dim:
            raise ValueError("expected [B,%d] flat states, got %s"
                             % (self.in_dim, tuple(x.shape)))
        g = x[:, :self.global_feats]
        fruit = x[:, self.global_feats:].reshape(-1, self.max_fruits, FRUIT_FEATS)
        mask = fruit[:, :, 2] > 0.0
        return g, fruit, mask

    def forward(self, x):
        g, fruit, fruit_mask = self._split_flat(x.float())
        cls = self.global_proj(g).unsqueeze(1) + self.cls_bias
        tok = self.fruit_proj(fruit)
        # type_norm is 0..1; radius mask prevents padding cherries ambiguity.
        type_idx = torch.clamp(torch.round(fruit[:, :, 3] * 10.0),
                               0, NUM_FRUIT_TYPES - 1).long()
        tok = tok + self.type_embed(type_idx)
        seq = torch.cat([cls, tok], dim=1)
        cls_valid = torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device)
        valid = torch.cat([cls_valid, fruit_mask], dim=1)
        encoded = self.encoder(seq, src_key_padding_mask=~valid)
        pooled = self.norm(encoded[:, 0])
        logits = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def infer_batch(self, vecs, device):
        self.eval()
        t = torch.as_tensor(vecs, dtype=torch.float32, device=device)
        logits, values = self.forward(t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs, values.cpu().numpy()

    @torch.no_grad()
    def infer(self, vec, device):
        self.eval()
        t = torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(t)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, float(value.item())


def build_net(cfg):
    """Construct the policy/value network selected by ``cfg['arch']``."""
    arch = str(cfg.get("arch", "mlp")).lower()
    K = int(cfg["K"])
    max_fruits = int(cfg["max_fruits"])
    boundary_features = bool(cfg.get("boundary_features", False))
    if arch in ("mlp", "flat", "flattened"):
        return PolicyValueNet(
            input_dim(max_fruits, boundary_features), K,
            hidden=int(cfg.get("hidden", 384)),
            n_layers=int(cfg.get("n_layers", 3)),
        )
    if arch in ("transformer", "object_transformer"):
        return TransformerPolicyValueNet(
            K=K,
            max_fruits=max_fruits,
            d_model=int(cfg.get("d_model", cfg.get("token_dim", 128))),
            n_heads=int(cfg.get("n_heads", 4)),
            n_layers=int(cfg.get("n_transformer_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            ff_mult=int(cfg.get("transformer_ff_mult", 4)),
            boundary_features=boundary_features,
        )
    raise ValueError("unknown network arch: %s" % arch)


def pick_device(prefer="mps"):
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
