import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset
from oxford_pet import load_dataset
from models.unet import UNet
from utils import dice_score
from models.resnet34_unet import ResNet34UNet

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True 
    print(f"Using device: {device}")

    if not os.path.exists(os.path.join(args.data_path, "annotations", "trainval.txt")):
        print(f"Dataset not found in {args.data_path}, downloading...")
        OxfordPetDataset.download(args.data_path)
        print("Download completed.")
    train_dataset = load_dataset(args.data_path, mode="train")
    val_dataset = load_dataset(args.data_path, mode="valid")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    if args.model_type == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model_type == "resnet34unet":
        model = ResNet34UNet(in_channels=3, n_classes=1).to(device)

    bceloss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_dice = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(images)
            bce_loss = bceloss(preds, masks)
            dice_loss = 1 - dice_score(preds, masks)
            loss = bce_loss + dice_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                preds = model(images)
                pred_mask = (preds > 0.5).float()
                val_dice += dice_score(pred_mask, masks)

        avg_val_dice = val_dice / len(val_loader)
        print(f"[Epoch {epoch+1}] Val Dice Score: {avg_val_dice:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs("../saved_models", exist_ok=True) 
            save_path = os.path.join("../saved_models", f"{args.model_type}_best.pth")
            torch.save(model.state_dict(), save_path)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'resnet34unet'], help='choose model type')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)