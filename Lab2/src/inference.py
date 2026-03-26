import argparse
import torch
from torch.utils.data import DataLoader
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from oxford_pet import load_dataset
from utils import dice_score

from tqdm import tqdm

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 抓 data
    test_dataset = load_dataset(args.data_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 選擇使用 unet or resnet34unet + unet
    if args.model_type == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model_type == "resnet34unet":
        model = ResNet34UNet(in_channels=3, n_classes=1).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model}")

    # Dice Score
    dice_total = 0.0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            preds = model(images)
            pred_mask = (preds > 0.5).float()
            dice_total += dice_score(pred_mask, masks)

    avg_dice = dice_total / len(test_loader)
    print(f"Average Dice Score on Test Set: {avg_dice:.4f}")
    return avg_dice

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'resnet34unet'], help='choose model type')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inference(args)