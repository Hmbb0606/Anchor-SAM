import torch
from torchmetrics import Metric

def custom_binary_iou(preds: torch.Tensor, target: torch.Tensor, threshold=0.5, smooth=1e-6) -> torch.Tensor:
    """
    手动计算二元分割的IoU (Jaccard Index)。
    """
    preds = (preds > threshold).float()
    target = target.float()
    preds = preds.view(-1)
    target = target.view(-1)
    intersection = (preds * target).sum()
    total = (preds + target).sum()
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


class CustomIoU(Metric):

    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.sum_iou += custom_binary_iou(preds, target)
        self.total += 1

    def compute(self):
        return self.sum_iou / self.total