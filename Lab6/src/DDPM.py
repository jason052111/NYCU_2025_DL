# Lab6: Conditional DDPM for i-CLEVR (CBAM U-Net + CFG)
# ---------------------------------------------------------------
# This single file can train, sample, and evaluate.
# What's new vs. your previous version:
# - Classifier-Free Guidance (CFG) at sampling time (--guidance)
# - Conditional dropout during training (--cond_drop)
# - Correct posterior std in p_sample (sigma fix)
# - GN made robust when channels not divisible by 8
#
# Examples
# Train (short finetune on existing ckpt):
#   python lab6_ddpm_minimal_cbam_cfg.py --data_root ./iclevr --train \
#       --epochs 6 --batch_size 128 --beta_schedule cosine --timesteps 250 \
#       --lr 1e-4 --cond_drop 0.1 --ckpt_path ./images/ddpm_ckpt.pt
# Sample + Evaluate with CFG:
#   python lab6_ddpm_minimal_cbam_cfg.py --sample --eval \
#       --ckpt_path ./images/ddpm_ckpt.pt --beta_schedule cosine --timesteps 250 \
#       --guidance 3.0

import os
import json
import math
import argparse
import csv
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image

# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_objects(objects_json_path):
    with open(objects_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def multi_labels_to_onehot(labels, mapping, num_classes=24):
    vec = torch.zeros(num_classes)
    for name in labels:
        if name in mapping:
            vec[mapping[name]] = 1.0
    return vec


def denorm_to_01(x):
    return (x.clamp(-1, 1) + 1.0) / 2.0

# ------------------------------
# Device helper
# ------------------------------

def report_device(tag=""):
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            print(f"[Device] {tag} CUDA available: True | index: {idx} | name: {name}")
        except Exception:
            print(f"[Device] {tag} CUDA available: True")
    else:
        print(f"[Device] {tag} CUDA available: False | using CPU")

# ------------------------------
# Dataset
# ------------------------------

class ICLEVRDataset(Dataset):
    def __init__(self, data_root, train_json, objects_json, image_size=64):
        super().__init__()
        self.data_root = data_root
        self.mapping = load_objects(objects_json)
        with open(train_json, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)  # dict: filename -> [labels]
        self.filenames = list(self.meta.keys())
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        labels = self.meta[fname]
        img_path = os.path.join(self.data_root, fname)
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)
        y = multi_labels_to_onehot(labels, self.mapping, num_classes=24)
        return x, y

# ------------------------------
# Diffusion helpers
# ------------------------------

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

# ------------------------------
# CBAM blocks
# ------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(in_channels // reduction_ratio, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = self.fc(self.global_avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_pool = self.fc(self.global_max_pool(x).view(b, c)).view(b, c, 1, 1)
        attention = self.sigmoid(avg_pool + max_pool)
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        pool_out = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(pool_out))
        return x * attention

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# ------------------------------
# Residual block with FiLM (time+cond) + robust GN
# ------------------------------

def _make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    g = min(max_groups, num_channels)
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch, use_cbam: bool = True):
        super().__init__()
        self.norm1 = _make_gn(in_ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = _make_gn(out_ch)
        self.act2  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(emb_ch, out_ch * 2)  # FiLM: scale, shift
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(self.act1(self.norm1(x)))
        scale, shift = self.emb_proj(emb).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.act2(h))
        h = h + self.skip(x)
        h = self.cbam(h)
        return h

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResidualBlock(in_ch, out_ch, emb_ch)
    def forward(self, x, emb):
        return self.block(self.pool(x), emb)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch, skip_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResidualBlock(out_ch + skip_ch, out_ch, emb_ch)
    def forward(self, x, skip, emb):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.block(x, emb)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNetCond(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, emb_dim: int = 256, num_classes: int = 24):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(num_classes, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.inc   = ResidualBlock(in_ch,   64,  emb_dim)
        self.down1 = Down(64,  128, emb_dim)
        self.down2 = Down(128, 256, emb_dim)
        self.down3 = Down(256, 512, emb_dim)
        self.down4 = Down(512, 512, emb_dim)
        self.up1 = Up(in_ch=512, out_ch=256, emb_ch=emb_dim, skip_ch=512)
        self.up2 = Up(in_ch=256, out_ch=128, emb_ch=emb_dim, skip_ch=256)
        self.up3 = Up(in_ch=128, out_ch= 64, emb_ch=emb_dim, skip_ch=128)
        self.up4 = Up(in_ch= 64, out_ch= 64, emb_ch=emb_dim, skip_ch= 64)
        self.out_norm = _make_gn(64)
        self.out_act  = nn.SiLU()
        self.outc = OutConv(64, out_ch)
    def forward(self, x, t, y):
        t_emb = self.time_mlp(timestep_embedding(t, self.time_mlp[0].in_features))
        y_emb = self.cond_mlp(y.float())
        emb   = t_emb + y_emb
        x1 = self.inc(x,   emb)
        x2 = self.down1(x1, emb)
        x3 = self.down2(x2, emb)
        x4 = self.down3(x3, emb)
        x5 = self.down4(x4, emb)
        x  = self.up1(x5, x4, emb)
        x  = self.up2(x,  x3, emb)
        x  = self.up3(x,  x2, emb)
        x  = self.up4(x,  x1, emb)
        x  = self.out_act(self.out_norm(x))
        out = self.outc(x)
        return out

# ------------------------------
# DDPM core (with sigma fix + CFG)
# ------------------------------

class DDPM(nn.Module):
    def __init__(self, model: nn.Module, image_size: int = 64, timesteps: int = 1000, beta_schedule: str = 'linear'):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.T = timesteps
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, self.T)
        elif beta_schedule == 'cosine':
            steps = torch.arange(self.T + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(((steps / self.T) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(1e-8, 0.999)
        else:
            raise ValueError('Unknown beta_schedule')
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_prev', torch.cat([torch.tensor([1.0], dtype=torch.float32, device=alphas.device), alphas_cumprod[:-1]]))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise

    def p_losses(self, x0, t, y):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        noise_pred = self.model(xt, t, y)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, x, t, y, guidance: float = 0.0):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        ac_t    = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        ac_prev = self.alphas_prev[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)

        if guidance and guidance > 0:
            eps_c = self.model(x, t, y)
            eps_u = self.model(x, t, torch.zeros_like(y))
            eps   = eps_u + guidance * (eps_c - eps_u)
        else:
            eps = self.model(x, t, y)

        mean = (1 / torch.sqrt(alpha_t)) * (x - betas_t / self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1) * eps)
        sigma = torch.sqrt(betas_t * (1 - ac_prev) / (1 - ac_t))  # <-- correct std
        noise = torch.randn_like(x) if (t > 0).any() else torch.zeros_like(x)
        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, batch_size, y, device, guidance: float = 0.0):
        x = torch.randn(batch_size, 3, self.image_size, self.image_size, device=device)
        for t in reversed(range(self.T)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor, y, guidance=guidance)
        return x

# ------------------------------
# Training & Evaluation
# ------------------------------

def train_loop(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report_device("main")
    set_seed(args.seed)
    report_device("train_loop")

    dataset = ICLEVRDataset(args.data_root, args.train_json, args.objects_json, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available(), drop_last=True)

    unet = UNetCond(in_ch=3, out_ch=3, emb_dim=256, num_classes=24)
    ddpm = DDPM(unet, image_size=args.image_size, timesteps=args.timesteps, beta_schedule=args.beta_schedule)
    ddpm.to(device)
    report_device("load_model")

    # Optionally resume
    if args.ckpt_path and os.path.isfile(args.ckpt_path):
        try:
            ckpt = torch.load(args.ckpt_path, map_location=device)
            ddpm.load_state_dict(ckpt['model'])
            print(f"[Resumed] {args.ckpt_path}")
        except Exception as e:
            print(f"[WARN] resume failed: {e}")

    opt = torch.optim.AdamW(ddpm.parameters(), lr=args.lr)

    ensure_dir(args.out_dir)
    log_csv = os.path.join(args.out_dir, "train_log.csv")
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["step", "epoch", "loss"])    

    global_step = 0
    ddpm.train()
    for epoch in range(args.epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # conditional dropout for CFG training
            if args.cond_drop > 0:
                drop = (torch.rand(y.size(0), 1, device=device) < args.cond_drop).float()
                y_in = y * (1 - drop)  # zero-out labels for some samples
            else:
                y_in = y

            t = torch.randint(0, ddpm.T, (x.size(0),), device=device).long()
            loss = ddpm.p_losses(x, t, y_in)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()

            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow([global_step, epoch, float(loss.item())])

            if global_step % args.log_every == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")
            global_step += 1

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, 'ddpm_ckpt.pt')
            torch.save({'model': ddpm.state_dict()}, ckpt_path)
            print(f"[Saved] {ckpt_path}")

    ckpt_path = os.path.join(args.out_dir, 'ddpm_ckpt.pt')
    torch.save({'model': ddpm.state_dict()}, ckpt_path)
    print(f"[Saved final] {ckpt_path}")

    try:
        import matplotlib.pyplot as plt
        steps, losses = [], []
        with open(log_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                steps.append(int(r["step"]))
                losses.append(float(r["loss"]))
        if len(steps) > 0:
            plt.figure()
            plt.plot(steps, losses)
            plt.xlabel("step")
            plt.ylabel("loss (MSE)")
            plt.title("Training Loss")
            plt.tight_layout()
            out_png = os.path.join(args.out_dir, "loss_curve.png")
            plt.savefig(out_png, dpi=160)
            plt.close()
            print(f"[Saved] {out_png}")
    except Exception as e:
        print(f"[WARN] plot loss failed: {e}")


def load_model(args, device):
    unet = UNetCond(in_ch=3, out_ch=3, emb_dim=256, num_classes=24)
    ddpm = DDPM(unet, image_size=args.image_size, timesteps=args.timesteps, beta_schedule=args.beta_schedule)
    ddpm.to(device)
    if args.ckpt_path and os.path.isfile(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location=device)
        ddpm.load_state_dict(ckpt['model'])
        print(f"[Loaded] {args.ckpt_path}")
    else:
        print("[WARN] No checkpoint loaded; sampling from an untrained model will look random.")
    ddpm.eval()
    return ddpm

@torch.no_grad()
def save_grid_for_conditions(ddpm: DDPM, label_sets: List[List[str]], mapping: Dict[str,int], out_path: str, device, nrow: int = 8, guidance: float = 0.0):
    conds = []
    for labels in label_sets:
        conds.append(multi_labels_to_onehot(labels, mapping).unsqueeze(0))
    Y = torch.cat(conds, dim=0).to(device)
    X = ddpm.sample(Y.size(0), Y, device, guidance=guidance)
    grid = make_grid(denorm_to_01(X.cpu()), nrow=nrow)
    save_image(grid, out_path)
    print(f"[Saved] {out_path}")
    return X, Y

@torch.no_grad()
def save_denoise_process(ddpm: DDPM, labels: List[str], mapping: Dict[str,int], steps: int, out_path: str, device, guidance: float = 0.0):
    B = 1
    Y = multi_labels_to_onehot(labels, mapping).unsqueeze(0).to(device)
    x = torch.randn(B, 3, ddpm.image_size, ddpm.image_size, device=device)
    frames = []
    t_indices = list(reversed(range(ddpm.T)))
    capture_ids = torch.linspace(0, ddpm.T-1, steps).long().tolist()
    for i, t in enumerate(t_indices):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        x = ddpm.p_sample(x, t_tensor, Y, guidance=guidance)
        if t in capture_ids or t == 0:
            frames.append(denorm_to_01(x.detach().cpu()))
    row = torch.cat(frames, dim=0)
    grid = make_grid(row, nrow=len(frames))
    save_image(grid, out_path)
    print(f"[Saved] {out_path}")

# ------------------------------
# Evaluator integration
# ------------------------------

def evaluate_generated(X: torch.Tensor, Y: torch.Tensor) -> float:
    if not torch.cuda.is_available():
        print("[Device] CUDA available: False | evaluator.py requires GPU (model .cuda()).")
        raise RuntimeError("CUDA not available; evaluator requires GPU.")
    try:
        from evaluator import evaluation_model
    except Exception as e:
        print("[ERROR] Cannot import evaluator.py. Make sure it's in the same directory.")
        raise e
    model = evaluation_model()
    X_cuda = X.cuda()
    Y_cuda = Y.cuda()
    acc = model.eval(X_cuda, Y_cuda)
    print(f"[Evaluator] accuracy = {acc:.4f}")
    return float(acc)

# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./iclevr')
    parser.add_argument('--train_json', type=str, default='./train.json')
    parser.add_argument('--test_json', type=str, default='./test.json')
    parser.add_argument('--new_test_json', type=str, default='./new_test.json')
    parser.add_argument('--objects_json', type=str, default='./objects.json')
    parser.add_argument('--out_dir', type=str, default='./images')

    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear','cosine'])

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=9999999)

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='')

    parser.add_argument('--process_steps', type=int, default=8)

    # NEW: CFG and conditional dropout
    parser.add_argument('--guidance', type=float, default=0.0, help='CFG scale (0=off)')
    parser.add_argument('--cond_drop', type=float, default=0.0, help='probability to drop condition during training')

    args = parser.parse_args()
    ensure_dir(args.out_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train:
        train_loop(args)

    if args.sample:
        mapping = load_objects(args.objects_json)
        ddpm = load_model(args, device)

        with open(args.test_json, 'r', encoding='utf-8') as f:
            test_list = json.load(f)
        with open(args.new_test_json, 'r', encoding='utf-8') as f:
            new_test_list = json.load(f)

        X_test, Y_test = save_grid_for_conditions(ddpm, test_list, mapping, os.path.join(args.out_dir, 'grid_test.png'), device, guidance=args.guidance)
        X_new, Y_new   = save_grid_for_conditions(ddpm, new_test_list, mapping, os.path.join(args.out_dir, 'grid_new_test.png'), device, guidance=args.guidance)

        process_labels = ["red sphere", "cyan cylinder", "cyan cube"]
        save_denoise_process(ddpm, process_labels, mapping, steps=args.process_steps, out_path=os.path.join(args.out_dir, 'denoise_process.png'), device=device, guidance=args.guidance)

        if args.eval:
            try:
                acc_test = evaluate_generated(X_test, Y_test)
            except Exception as e:
                print(f"[Eval Error - test.json] {e}")
                acc_test = float('nan')
            try:
                acc_new  = evaluate_generated(X_new, Y_new)
            except Exception as e:
                print(f"[Eval Error - new_test.json] {e}")
                acc_new = float('nan')
            with open(os.path.join(args.out_dir, 'accuracy.txt'), 'w') as f:
                f.write(f"test.json accuracy: {acc_test}\n")
                f.write(f"new_test.json accuracy: {acc_new}\n")
            print("[Saved] accuracy.txt")

    if not args.train and not args.sample:
        print("Nothing to do. Add --train and/or --sample flags. See header for examples.")

if __name__ == '__main__':
    main()
