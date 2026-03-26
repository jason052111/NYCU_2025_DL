import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
import shutil


# -----------------------------
# TODO2 step1-4: training loop
# -----------------------------
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim, self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.writer = None  # 只有 log 時才會被 init_logging 設定

    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        self.optim.zero_grad(set_to_none=True)

        total_loss = 0.0
        accum = max(1, self.args.accum_grad)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, image in pbar:
            x = image.to(self.args.device, non_blocking=True)

            # forward (logits: BxNx(V+1), z_indices: BxN)
            logits, z_indices = self.model(x)

            # CE over flattened tokens
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                z_indices.reshape(-1),
            )

            # 累積梯度：為了保總梯度一致，先除以 accum
            (loss / accum).backward()
            total_loss += loss.item()

            # 每 accum step 更新一次
            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

            pbar.set_description(f"(train) Epoch {epoch} | loss {loss.item():.4f}", refresh=False)

        if self.scheduler is not None:
            # 這裡用 epoch 粗略 step 一次（簡單就好）
            self.scheduler.step()

        avg_loss = total_loss / len(train_loader)
        if self.args.log and self.writer is not None:
            import wandb
            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            wandb.log({"Loss/train": avg_loss, "epoch": epoch})
        return avg_loss

    @torch.no_grad()
    def eval_one_epoch(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for step, image in pbar:
            x = image.to(self.args.device, non_blocking=True)
            logits, z_indices = self.model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                z_indices.reshape(-1),
            )
            total_loss += loss.item()
            pbar.set_description(f"(val)   Epoch {epoch} | loss {loss.item():.4f}", refresh=False)

        avg_loss = total_loss / len(val_loader)
        if self.args.log and self.writer is not None:
            import wandb
            self.writer.add_scalar("Loss/val", avg_loss, epoch)
            wandb.log({"Loss/val": avg_loss, "epoch": epoch})
        return avg_loss

    def configure_optimizers(self):
        # 想法不變：Adam 就好，參數跟你接近；scheduler 簡單一點可有可無
        opt = torch.optim.Adam(self.model.parameters(),
                               lr=self.args.learning_rate,
                               betas=(0.9, 0.96))
        sch = None  # 保持樸素
        return opt, sch

    def _ensure_run_dirs(self):
        run_dir = os.path.join("transformer_checkpoints", f"{self.args.run_name}-{self.args.run_id}")
        os.makedirs(run_dir, exist_ok=True)
        run_dir2 = os.path.join(self.args.ckpt_dir, f"{self.args.run_name}-{self.args.run_id}")
        os.makedirs(run_dir2, exist_ok=True)
        return run_dir, run_dir2

    def save_checkpoint(self, epoch, checkpoint_name=None):
        if checkpoint_name is None:
            checkpoint_name = f"epoch_{epoch}.pt"

        run_dir, run_dir2 = self._ensure_run_dirs()

        # 只存 transformer 權重（評分會用這個）
        torch.save(self.model.transformer.state_dict(),
                   os.path.join(run_dir, checkpoint_name))
        print(f"[save] transformer -> {os.path.join(run_dir, checkpoint_name)}")

        # 若要完整重訓，就連同整個 MaskGit + optimizer 一起存
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "last_epoch": epoch,
            "args": self.args
        }, os.path.join(run_dir2, checkpoint_name))
        print(f"[save] maskgit(full) -> {os.path.join(run_dir2, checkpoint_name)}")

    def load_checkpoint(self, checkpoint_path, val_loader):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt["state_dict"])
        self.optim.load_state_dict(ckpt["optimizer"])
        self.args.start_from_epoch = int(ckpt.get("last_epoch", 0))
        # 把學習率等從檔案帶回來（保守點）
        self.args.learning_rate = float(getattr(ckpt.get("args", self.args), "learning_rate", self.args.learning_rate))
        self.args.run_id = getattr(ckpt.get("args", self.args), "run_id", self.args.run_id)
        print(f"[load] {self.args.run_id} from {checkpoint_path} (epoch {self.args.start_from_epoch})")

        # 驗證一次看看
        prev = self.args.log
        self.args.log = False
        best_val = self.eval_one_epoch(val_loader, self.args.start_from_epoch)
        self.args.log = prev
        return best_val

    def init_logging(self):
        from torch.utils.tensorboard import SummaryWriter
        import wandb

        if not self.args.resume_path:
            self.args.run_id = wandb.util.generate_id()

        wandb.init(project="MaskGit", config=self.args, id=self.args.run_id, resume="allow")
        self.args.run_name = wandb.run.name

        self.writer = SummaryWriter(f"runs/{self.args.run_name}-{self.args.run_id}")
        self._ensure_run_dirs()

    def save_model_to_wandb(self, epoch):
        if not self.args.log:
            return
        try:
            import wandb
            run_dir = os.path.join("transformer_checkpoints", f"{self.args.run_name}-{self.args.run_id}")
            src = os.path.join(run_dir, "best_model.pt")
            if os.path.isfile(src):
                tmp_dir = f"tmp_{self.args.run_name}"
                os.makedirs(os.path.join(tmp_dir, "models"), exist_ok=True)
                dst = os.path.join(tmp_dir, "models", f"{self.args.run_name}-best-transformer.pt")
                shutil.copyfile(src, dst)
                wandb.save(os.path.abspath(dst), base_path=os.path.abspath(tmp_dir))
                print("[wandb] saved best transformer")
        except Exception as e:
            print(f"[wandb] save failed: {e}")

    def save_tensorboard_to_wandb(self):
        if not self.args.log:
            return
        try:
            import wandb
            wandb.save(os.path.abspath(f"runs/{self.args.run_name}-{self.args.run_id}"),
                       base_path=os.path.abspath("runs"))
            print("[wandb] saved tensorboard logs")
        except Exception as e:
            print(f"[wandb] save tb failed: {e}")

    def finish_training(self):
        if self.args.log and self.writer is not None:
            import wandb
            self.writer.close()
            wandb.finish()
        # 清臨時資料夾（存在才刪）
        tmp_dir = f"tmp_{self.args.run_name}"
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    # dataset
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path',   type=str, default="./cat_face/val/",   help='Validation Dataset Path')
    # io
    parser.add_argument('--ckpt-dir', type=str, default='./checkpoints/', help='Path to checkpoint.')
    parser.add_argument('--device',   type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size',  type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial',     type=float, default=1.0, help='Portion of dataset to use.')
    parser.add_argument('--accum-grad',  type=int, default=10, help='Gradient accumulation steps.')

    # hparams
    parser.add_argument('--epochs', type=int, default=300, help='Total epochs.')
    parser.add_argument('--save-per-epoch', type=int, default=5, help='Save ckpt every N epochs.')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Start epoch.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='(unused)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='LR.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Config for Transformer+VQGAN')

    # logging
    parser.add_argument('--log', action='store_true', help='Use wandb/tensorboard')
    parser.add_argument('--run-id', type=str, default="", help='Run ID for wandb')
    parser.add_argument('--resume-path', type=str, default=None, help='Resume from full MaskGit ckpt')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              drop_last=True,
                              pin_memory=True,
                              shuffle=True)

    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)

    # -----------------------------
    # TODO2 step1-5: training flow
    # -----------------------------
    if args.resume_path:
        best_val_loss = train_transformer.load_checkpoint(args.resume_path, val_loader)
    else:
        best_val_loss = float('inf')

    if train_transformer.args.log:
        train_transformer.init_logging()

    for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
        tr_loss = train_transformer.train_one_epoch(train_loader, epoch)
        va_loss = train_transformer.eval_one_epoch(val_loader, epoch)

        # 週期性保存
        if args.save_per_epoch > 0 and (epoch % args.save_per_epoch == 0):
            train_transformer.save_checkpoint(epoch)

        # 最佳模型（給 inpainting 用）
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            train_transformer.save_checkpoint(epoch, "best_model.pt")

    train_transformer.save_model_to_wandb(train_transformer.args.epochs)
    train_transformer.save_tensorboard_to_wandb()
    train_transformer.finish_training()
