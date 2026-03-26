import os
import math
import yaml
import numpy as np
import torch
import torch.nn as nn

from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


# -------------------------------------------
# MaskGIT (VQGAN + Bidirectional Transformer)
# -------------------------------------------
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # stage1 (fixed)
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])

        # stage2 (trainable)
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

        # misc / schedule
        self.num_image_tokens   = configs['num_image_tokens']            # 16*16
        self.mask_token_id      = configs['num_codebook_vectors']        # 第 (K+1) 個 id 當作 mask
        self.choice_temperature = configs.get('choice_temperature', 1.0)
        self.gamma              = self.gamma_func(configs.get('gamma_type', 'cosine'))

    # -----------------------
    # checkpoint helpers
    # -----------------------
    def load_transformer_checkpoint(self, load_ckpt_path: str):
        if load_ckpt_path and os.path.isfile(load_ckpt_path):
            self.transformer.load_state_dict(torch.load(load_ckpt_path, map_location='cpu'))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path'], map_location='cpu'), strict=True)
        model.eval()
        return model

    # -------------------------------------------------
    # TODO2 step1-1: image -> latent tokens (z_q, idx)
    # -------------------------------------------------
    @torch.no_grad()
    def encode_to_z(self, x: torch.Tensor):
        # VQGAN.encode 回傳: (z_q, indices, q_loss)
        z_q, idx, _ = self.vqgan.encode(x)
        idx = idx.view(x.size(0), -1).contiguous().long()   # (B, 256)
        return z_q, idx

    # -------------------------------------------------
    # TODO2 step1-2: scheduling function generator
    # -------------------------------------------------
    def gamma_func(self, mode: str = "cosine"):
        # 給一個 ratio \in [0,1)，回一個 mask 比例 (0,1]
        mm = (mode or "cosine").lower()
        if mm == "linear":
            return lambda r: 1.0 - float(r)
        elif mm == "cosine":
            # 早期保留多一點 mask，後期快速收斂
            return lambda r: float(np.cos(float(r) * np.pi * 0.5))
        elif mm == "square":
            # 跟你原本想法一致：後段掉得更快（但換個寫法）
            return lambda r: float((1.0 - float(r)) * (1.0 - float(r)))
        else:
            # 預設就線性
            return lambda r: 1.0 - float(r)

    # -------------------------------------------------
    # TODO2 step1-3: training forward
    # -------------------------------------------------
    def forward(self, x: torch.Tensor):
        # 1) 影像 token 化
        _, gt_idx = self.encode_to_z(x)          # (B, N)

        # 2) 隨機抽 ratio，再用 gamma 轉成 mask 比例
        r = np.random.rand()
        p = self.gamma(r)

        # 3) 依比例做隨機 mask（每個 token 獨立擲一個 U(0,1)）
        rnd  = torch.rand_like(gt_idx, dtype=torch.float32)
        msk  = rnd < p                             # bool, True=mask
        tok  = gt_idx.clone()
        tok[msk] = self.mask_token_id              # 蓋上特別 token

        # 4) 丟進 transformer
        logits = self.transformer(tok)             # (B, N, K+1)

        return logits, gt_idx                      # 交叉熵用 logits vs. gt

    # -------------------------------------------------
    # TODO3 step1-1: one-step iterative decoding
    # -------------------------------------------------
    @torch.no_grad()
    def inpainting(self,
                   z_indices: torch.Tensor,
                   mask_bc: torch.Tensor,
                   ratio: float,
                   mask_func: str = None):
        """
        單一步驟的 parallel 解碼：
        - 先把當前被 mask 的位置換成 mask_token_id
        - transformer 出 logits -> softmax
        - 取每個 token 的 argmax 與對應機率
        - 機率 + 退火 gumbel 噪聲形成 confidence
        - 依排程比例決定下一輪仍要 mask 的位置（低信心者優先）
        回傳: (更新後的 token, 下一輪的 mask)
        """

        # 1) 準備輸入（masked 位置替換成 mask_token_id）
        inp = z_indices.clone()
        inp[mask_bc] = self.mask_token_id

        # 2) 預測分佈
        logits = self.transformer(inp)                 # (B, N, K+1)
        probs  = torch.softmax(logits, dim=-1)

        # 3) 取最大機率的類別 & 機率
        pick_prob, pick_idx = probs.max(dim=-1)        # (B, N)

        # 4) 未被 mask 的位置不動（給它 inf，等等排序就不會被再次選中）
        pick_prob = pick_prob.masked_fill(~mask_bc, float('inf'))

        # 5) 退火的 Gumbel 噪聲（跟你相同概念，用另一種寫法）
        #    g = -log(-log(U))，這裡用 torch 的簡易生成
        g = -torch.empty_like(pick_prob).exponential_().log()
        temp = self.choice_temperature * (1.0 - float(ratio))
        confidence = pick_prob + temp * g

        # 6) 更新 token：只有被 mask 的格子才採用 transformer 的預測
        new_tokens = torch.where(mask_bc, pick_idx.to(z_indices.dtype), z_indices)

        # 7) 依排程決定下一輪還要保留的 mask 數量（低信心者優先）
        sched = self.gamma if mask_func is None else self.gamma_func(mask_func)
        keep_ratio = sched(ratio)  # 下一輪仍保留的「mask 比例」

        # 每個 batch 計算要保留多少個 mask（用目前 true 的數量）
        cur_mask_cnt = mask_bc.sum(dim=1).to(torch.float32)        # (B,)
        keep_num     = (keep_ratio * cur_mask_cnt).floor().to(torch.long)  # (B,)

        # 對 confidence 升冪排序（小→大：越不確定、越該繼續 mask）
        _, order = torch.sort(confidence, dim=1, descending=False)  # (B, N)

        next_mask = torch.zeros_like(mask_bc)
        # 用一個土方法逐 batch 生成下一輪的 mask，不求最潮，但直觀
        B = order.size(0)
        for b in range(B):
            k = int(keep_num[b].item())
            if k > 0:
                idxs = order[b, :k]          # 取最不確定的前 k 個
                next_mask[b].scatter_(0, idxs, True)

        return new_tokens, next_mask


__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
