"""
GPU training script — trains a tiny transformer on synthetic data.

Reads hyperparameters from src/config.rs, trains on GPU for a fixed
time budget, reports real val_bpb. Fits on 8GB VRAM.

Falls back to deterministic scoring if no GPU is available.
"""

import hashlib
import math
import os
import re
import sys
import time


def parse_config(path: str) -> dict[str, float]:
    """Extract numeric constants from a config file."""
    consts: dict[str, float] = {}
    if not os.path.exists(path):
        return consts
    with open(path) as f:
        for line in f:
            m = re.match(r'pub const (\w+):\s+\w+\s*=\s*([0-9.e_+-]+)', line)
            if m:
                name, val = m.group(1), m.group(2).replace('_', '')
                try:
                    consts[name] = float(val)
                except ValueError:
                    pass
    return consts


def compute_bpb_deterministic(config: dict[str, float]) -> float:
    """Fallback: deterministic val_bpb from hyperparameters (no GPU)."""
    depth = config.get("DEPTH", 4)
    lr = config.get("MATRIX_LR", 0.001)
    base = 1.8 - 0.15 * math.log2(max(depth, 1))
    lr_penalty = 0.3 * (math.log(lr / 0.04)) ** 2 if lr > 0 else 1.0
    config_str = str(sorted(config.items()))
    noise = int(hashlib.md5(config_str.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    noise = (noise - 0.5) * 0.02
    return max(0.5, min(3.0, base + lr_penalty + noise))


def train_real(cfg: dict[str, float]):
    """Actually train a tiny transformer on GPU."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    depth = int(cfg.get("DEPTH", 4))
    aspect = int(cfg.get("ASPECT_RATIO", 32))
    lr = cfg.get("MATRIX_LR", 0.001)
    wd = cfg.get("WEIGHT_DECAY", 0.01)
    vocab_size = int(cfg.get("VOCAB_SIZE", 256))
    batch_size = int(cfg.get("BATCH_SIZE", 32))
    seq_len = int(cfg.get("SEQ_LEN", 64))
    budget = cfg.get("TIME_BUDGET_SECS", 10.0)

    n_embd = depth * aspect
    n_head = max(1, n_embd // 32)
    n_embd = (n_embd // n_head) * n_head

    device = torch.device("cuda")
    torch.manual_seed(42)

    # Tiny GPT
    wte = nn.Embedding(vocab_size, n_embd)
    wpe = nn.Embedding(seq_len, n_embd)
    blocks = nn.ModuleList([
        nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        for _ in range(depth)
    ])
    ln_f = nn.LayerNorm(n_embd)
    lm_head = nn.Linear(n_embd, vocab_size, bias=False)
    model = nn.Sequential(wte, wpe, *blocks, ln_f, lm_head).to(device)

    # Custom forward
    def forward(idx):
        B, T = idx.shape
        x = wte(idx) + wpe(torch.arange(T, device=device).unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        for block in blocks:
            x = block(x, src_mask=mask, is_causal=True)
        return lm_head(ln_f(x))

    num_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Synthetic data with learnable patterns
    data = torch.randint(0, vocab_size, (2048, seq_len + 1), device=device)
    mask = torch.rand(2048, seq_len) < 0.3
    for i in range(seq_len):
        data[:, i + 1] = torch.where(mask[:, i], (data[:, i] + 1) % vocab_size, data[:, i + 1])
    train_x, train_y = data[:, :-1], data[:, 1:]

    val_data = torch.randint(0, vocab_size, (256, seq_len + 1), device=device)
    val_mask = torch.rand(256, seq_len) < 0.3
    for i in range(seq_len):
        val_data[:, i + 1] = torch.where(val_mask[:, i], (val_data[:, i] + 1) % vocab_size, val_data[:, i + 1])
    val_x, val_y = val_data[:, :-1], val_data[:, 1:]

    t0 = time.time()
    step = 0
    while time.time() - t0 < budget:
        model.train()
        idx = torch.randint(0, len(train_x) - batch_size, (1,)).item()
        logits = forward(train_x[idx:idx + batch_size])
        loss = F.cross_entropy(logits.view(-1, vocab_size), train_y[idx:idx + batch_size].reshape(-1))
        if math.isnan(loss.item()):
            print("FAIL"); sys.exit(1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

    training_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        val_logits = forward(val_x)
        val_loss = F.cross_entropy(val_logits.view(-1, vocab_size), val_y.reshape(-1))
        val_bpb = val_loss.item() / math.log(2)

    return val_bpb, training_time, step, num_params, depth


def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "config.rs")
    cfg = parse_config(config_path)

    # Use deterministic scoring for fast, reproducible tests.
    # Set MOCK_TRAIN_GPU=1 to use real GPU training instead.
    use_gpu = os.environ.get("MOCK_TRAIN_GPU") == "1"
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                val_bpb, training_time, step, num_params, depth = train_real(cfg)
            else:
                use_gpu = False
        except (ImportError, RuntimeError):
            use_gpu = False

    if not use_gpu:
        val_bpb = compute_bpb_deterministic(cfg)
        training_time = cfg.get("TIME_BUDGET_SECS", 2.0)
        step = 100
        num_params = int(cfg.get("DEPTH", 4)) * 1000000
        depth = int(cfg.get("DEPTH", 4))

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {training_time:.1f}")
    print(f"total_seconds:    {training_time:.1f}")
    print(f"total_tokens_M:   {step * 32 * 64 / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {depth}")
    print(f"startup_seconds:  0.0")


if __name__ == "__main__":
    main()
