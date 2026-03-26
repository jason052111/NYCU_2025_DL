import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import argparse
from utils import LoadTestData, LoadMaskData
from torch.utils.data import Dataset, DataLoader
from torchvision import utils as vutils
import os
from models import MaskGit as VQGANTransformer
import yaml
import torch.nn.functional as F


class MaskGIT:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.model.load_transformer_checkpoint(args.load_transformer_ckpt_path)
        self.model.eval()
        self.total_iter = args.total_iter
        self.mask_func = args.mask_func
        self.sweet_spot = args.sweet_spot
        self.device = args.device
        self.prepare()

    @staticmethod
    def prepare():
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("mask_scheduling", exist_ok=True)
        os.makedirs("imga", exist_ok=True)

    # ------------------------------
    # TODO3 step1-1: total iteration
    # ------------------------------
    # mask_b: 初始遮罩 (True=要預測)
    def inpainting(self, image, mask_b, i):
        # 視覺化用 buffer：都放 CPU，省 GPU 記憶體
        maska = torch.zeros(self.total_iter, 3, 16, 16)          # 每步的 latent mask
        imga  = torch.zeros(self.total_iter + 1, 3, 64, 64)      # 每步 decode 的影像

        # 還原到可視化範圍
        mean = torch.tensor([0.4868, 0.4341, 0.3844], device=self.device).view(3, 1, 1)
        std  = torch.tensor([0.2620, 0.2527, 0.2543], device=self.device).view(3, 1, 1)

        # 第 0 張：被遮的輸入（只做展示）
        ori = (image[0] * std) + mean
        imga[0] = ori.detach().cpu()

        with torch.no_grad():
            # 取 token（B=1 假設）
            _, z_idx0 = self.model.encode_to_z(image)      # (1, 256)
            z_work = z_idx0.clone().to(self.device)        # 當前 token 狀態
            cur_mask = mask_b.to(self.device)              # 當前 mask（True=還要預測）

            # 逐步解碼
            last_vis = None
            for step in range(self.total_iter):
                # sweet spot 到就停
                if step == self.sweet_spot:
                    break

                # 0~1 的相對步數
                ratio = float(step + 1) / float(self.total_iter)

                # 呼叫模型的一步推理（會回傳：更新後 token、下一輪 mask）
                z_work, cur_mask = self.model.inpainting(
                    z_work, cur_mask, ratio, self.mask_func
                )

                # ---- 視覺化 latent mask（白=保留，黑=仍 mask）
                m16 = cur_mask.view(1, 16, 16).float().cpu()  # 1x16x16
                # 建個 3x16x16 的「白板」，mask 為真處畫黑
                m_img = torch.ones(3, 16, 16)
                pos = (m16[0] > 0.5).nonzero(as_tuple=False)  # N x 2
                if pos.numel() > 0:
                    m_img[:, pos[:, 0], pos[:, 1]] = 0.0
                maska[step] = m_img

                # ---- decode 成 64x64 圖（注意 embedding -> (B,H,W,C) -> (B,C,H,W)）
                #   這裡不用太聰明，直接硬塞形狀
                shape = (1, 16, 16, 256)
                zq = self.model.vqgan.codebook.embedding(z_work.long()).view(shape)
                zq = zq.permute(0, 3, 1, 2).contiguous()
                out = self.model.vqgan.decode(zq)            # (1,3,64,64)
                vis = (out[0] * std) + mean                  # 反標準化（在 GPU）
                last_vis = vis
                imga[step + 1] = vis.detach().cpu()

            # 只把 sweet spot 的輸出丟進 FID 的資料夾
            if last_vis is None:
                # 沒跑迭代就寫入輸入圖，避免空檔
                last_vis = ori
            vutils.save_image(last_vis, os.path.join("test_results", f"image_{i:03d}.png"), nrow=1)

            # demo：把整條過程也各存一張大圖
            vutils.save_image(maska, os.path.join("mask_scheduling", f"test_{i}.png"), nrow=10)
            vutils.save_image(imga,  os.path.join("imga",            f"test_{i}.png"), nrow=7)


class MaskedImage:
    def __init__(self, args):
        mi_ori = LoadTestData(root=args.test_maskedimage_path, partial=args.partial)
        self.mi_ori = DataLoader(
            mi_ori,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=False,
        )
        mask_ori = LoadMaskData(root=args.test_mask_path, partial=args.partial)
        self.mask_ori = DataLoader(
            mask_ori,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=False,
        )
        self.device = args.device

    def get_mask_latent(self, mask):
        # 64x64 -> 32x32 -> 16x16 的粗暴降採樣（跟你想法相同）
        m1 = F.avg_pool2d(mask, kernel_size=2, stride=2)
        m2 = F.avg_pool2d(m1,   kernel_size=2, stride=2)

        # 只允許 0/1，非 1 的都歸零
        m2 = (m2 == 1).float()

        # 取其中一個通道 -> 攤平成 256
        tok = m2[0, 0].flatten()              # (256,)
        tok = tok.unsqueeze(0)                 # (1,256)

        # True = 要預測（這裡沿用你原本「0 被視為 mask」的邏輯）
        mb = torch.zeros_like(tok, dtype=torch.bool, device=self.device)
        mb |= (tok == 0)
        return mb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT for Inpainting")
    parser.add_argument('--device', type=str, default="cuda", help='Which device the testing is on.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing.')
    parser.add_argument('--partial', type=float, default=1.0, help='Portion of dataset to use.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for MaskGIT')

    # TODO3 step1-2: 路徑/參數你已經給了，我保持一樣
    parser.add_argument('--load-transformer-ckpt-path', type=str, default='./transformer_checkpoints/best_model.pt', help='path to transformer ckpt')

    # dataset path
    parser.add_argument('--test-maskedimage-path', type=str, default='./cat_face/masked_image', help='Path to testing image dataset.')
    parser.add_argument('--test-mask-path', type=str, default='./cat_face/mask64', help='Path to testing mask dataset.')

    # MVTM 參數
    parser.add_argument('--sweet-spot', type=int, default=8,   help='best step in total iteration')
    parser.add_argument('--total-iter', type=int, default=8,   help='total steps for scheduling')
    parser.add_argument('--mask-func',  type=str, default='cosine', help='mask scheduling function')

    args = parser.parse_args()

    t = MaskedImage(args)
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    maskgit = MaskGIT(args, MaskGit_CONFIGS)
    print(f"Start Inpainting using {args.mask_func} mask scheduling function")

    idx = 0
    for image, mask in zip(t.mi_ori, t.mask_ori):
        image = image.to(device=args.device)
        mask  = mask.to(device=args.device)
        mb = t.get_mask_latent(mask)
        maskgit.inpainting(image, mb, idx)
        idx += 1
