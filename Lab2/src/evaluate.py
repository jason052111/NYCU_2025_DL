import torch
from utils import dice_score

def evaluate(net, data, device):
    net.eval()  
    dice_total = 0.0
    with torch.no_grad():
        for batch in data:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            preds = net(images)
            pred_mask = (preds > 0.5).float()
            dice_total += dice_score(pred_mask, masks)

    return dice_total / len(data)
