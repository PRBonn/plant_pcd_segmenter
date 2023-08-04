import torch
import torch.nn as nn


class OffsetLoss(nn.Module):
    def __init__(self, cfg):
        super(OffsetLoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()

    def offset_loss(self, pred_offsets, offset_labels):
        valid_mask = ~(offset_labels.isnan().all(-1))
        
        loss = self.l1_loss(pred_offsets[valid_mask], offset_labels[valid_mask])
        return loss

    def forward(self, pred_offsets, offset_labels):
        loss = self.offset_loss(pred_offsets, offset_labels)
        return loss


class ConfidIoULoss(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ConfidIoULoss, self).__init__()
        if "min_confid_thres" in self.cfg.keys():
            self.min_confid_thres = self.cfg["min_confid_thres"]
        else:
            self.min_confid_thres = 0.4

        if "loss_type" not in cfg.keys() or cfg["loss_type"] == "l2":
            self.loss_fn = torch.nn.MSELoss(reduction="none")
        elif cfg["loss_type"] == "l1":
            self.loss_fn = torch.nn.L1Loss(reduction="none")
        else:
            raise ValueError(
                "Loss type {} is not implemented.".format(cfg["loss_type"])
            )

    def confid_iou_loss(self, pred_batch, gt_ious):
        # filter out -1 fill values
        valid_mask = torch.logical_and(gt_ious > 0, pred_batch > 0)

        tolerant_loss = True
        if tolerant_loss:
            # mask out points with gt < thres and pred < thres (it is enough if the prediction for those points is below thres)
            masked_out = torch.logical_and(
                gt_ious < self.min_confid_thres, pred_batch < self.min_confid_thres
            )
            complete_mask = torch.logical_and(valid_mask, ~masked_out)
        else:
            complete_mask = valid_mask

        loss = self.loss_fn(pred_batch[complete_mask], gt_ious[complete_mask])

        penalize_overestim_w = 2
        if penalize_overestim_w > 0:
            overestim_mask = pred_batch[complete_mask] > gt_ious[complete_mask]
            loss[overestim_mask] *= penalize_overestim_w

        loss = loss.mean()

        return loss

    def forward(self, pred, gt_ious):
        loss = self.confid_iou_loss(pred, gt_ious)

        return loss
