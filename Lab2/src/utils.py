def dice_score(pred_mask, gt_mask, eps=1e-6 ):
# def dice_score(pred_mask, gt_mask ):

    pred_mask = pred_mask.float()   # GPT
    gt_mask = gt_mask.float()       # GPT

    pred_flat = pred_mask.view(-1)
    gt_flat = gt_mask.view(-1)

    intersection = (pred_flat * gt_flat).sum()
    dice = (2. * intersection + eps) / (pred_flat.sum() + gt_flat.sum() + eps)   # GPT add eps

    return dice.item()


