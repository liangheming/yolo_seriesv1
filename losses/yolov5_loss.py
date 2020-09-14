import torch
from torch import nn
from losses.commons import BoxSimilarity


class YOLOv5Builder(object):
    def __init__(self, ratio_thresh=4., expansion_bias=0.5):
        super(YOLOv5Builder, self).__init__()
        self.ratio_thresh = ratio_thresh
        self.expansion_bias = expansion_bias

    def __call__(self, predicts, targets, input_anchors):
        """
        :param predicts: list(predict) [bs,anchor_per_grid,ny,nx,output(num_cls+5)]
        :param targets: [gt_num,7] (bs_id,weight,label_idx,xc,yc,w,h){normalized w,h}
        :param input_anchors:[layer_num,anchor_per_grid,2(w,h)] {grid_scale w,h}
        :return:
        """
        device = targets.device
        num_layer, anchor_per_grid = input_anchors.shape[:2]
        num_gt = targets.shape[0]
        target_weights_cls, target_box, target_indices, target_anchors = list(), list(), list(), list()
        gain = torch.ones(8, device=device).float()
        # anchor_per_grid,gt_num
        anchor_idx = torch.arange(anchor_per_grid, device=device).float().view(anchor_per_grid, 1).repeat(1, num_gt)
        # [anchor_per_grid,num_gt,8](bs_idx,weights,label_idx,xc,yc,w,h,anchor_idx)
        targets = torch.cat((targets.repeat(anchor_per_grid, 1, 1), anchor_idx[:, :, None]), dim=2)
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1]], device=device).float() * self.expansion_bias
        for i in range(num_layer):
            # [anchor_per_grid,2]
            anchors = input_anchors[i]
            gain[3:7] = torch.tensor(predicts[i].shape)[[3, 2, 3, 2]]
            t = targets * gain
            if num_gt:
                r = t[:, :, 5:7] / anchors[:, None, :]
                valid_idx = torch.max(r, 1. / r).max(2)[0] < self.ratio_thresh
                t = t[valid_idx]
                gxy = t[:, 3:5]
                gxy_flip = gain[3:5] - gxy
                j, k = ((gxy % 1. < self.expansion_bias) & (gxy > 1.)).T
                l, m = ((gxy_flip % 1. < self.expansion_bias) & (gxy_flip > 1.)).T
                gain_valid_idx = torch.stack([torch.ones_like(j), j, k, l, m])
                t = t.repeat((5, 1, 1))[gain_valid_idx]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[gain_valid_idx]
            else:
                t = targets[0]
                offsets = 0

            b, c, a = t[:, [0, 2, 7]].long().T
            w = t[:, 1]
            gt_xy = t[:, 3:5]
            gt_wh = t[:, 5:7]
            gij = (gt_xy - offsets).long()
            gi, gj = gij.T

            target_indices.append((b, a, gj, gi))
            target_box.append(torch.cat((gt_xy - gij, gt_wh), dim=1))
            target_anchors.append(anchors[a])
            target_weights_cls.append((w, c))

        return target_weights_cls, target_box, target_indices, target_anchors


class YOLOv5LossOriginal(object):
    def __init__(self,
                 ratio_thresh=4.,
                 expansion_bias=0.5,
                 layer_balance=None,
                 cls_pw=1.0,
                 obj_pw=1.0,
                 iou_type="ciou",
                 coord_type="xywh",
                 iou_ratio=1.0,
                 iou_weights=0.05,
                 cls_weights=0.5,
                 obj_weights=1.0):
        super(YOLOv5LossOriginal, self).__init__()
        if layer_balance is None:
            layer_balance = [4.0, 1.0, 0.4]
        self.layer_balance = layer_balance
        self.iou_ratio = iou_ratio
        self.iou_weights = iou_weights
        self.cls_weights = cls_weights
        self.obj_weights = obj_weights
        self.expansion_bias = expansion_bias
        self.box_similarity = BoxSimilarity(iou_type, coord_type)
        self.target_builder = YOLOv5Builder(ratio_thresh, expansion_bias)
        self.cls_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(data=[cls_pw]))
        self.obj_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(data=[obj_pw]))

    def __call__(self, predicts, targets, input_anchors):
        device = targets.device
        if self.cls_bce.pos_weight.device != device:
            self.cls_bce.to(device)
        if self.obj_bce.pos_weight.device != device:
            self.obj_bce.to(device)
        loss_cls, loss_box, loss_obj = torch.zeros(1, device=device), \
                                       torch.zeros(1, device=device), \
                                       torch.zeros(1, device=device)
        targets_weights_cls, targets_box, targets_indices, targets_anchors = \
            self.target_builder(predicts, targets, input_anchors)
        num_cls = predicts[0].shape[-1] - 5
        num_bs = predicts[0].shape[0]
        match_num = 0
        layer_num = len(predicts)

        for i, pi in enumerate(predicts):
            b, a, gj, gi = targets_indices[i]
            target_obj = torch.zeros_like(pi[..., 0], device=device)
            n = len(b)
            if n:
                match_num += n
                ps = pi[b, a, gj, gi]

                pxy = ps[:, :2].sigmoid() * 2. - self.expansion_bias
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * targets_anchors[i]
                pbox = torch.cat((pxy, pwh), dim=1).to(device)
                box_sim = self.box_similarity(pbox, targets_box[i])
                loss_box += (1.0 - box_sim).mean()

                target_obj[b, a, gj, gi] = \
                    (1.0 - self.iou_ratio) + self.iou_ratio * box_sim.detach().clamp(0).type(target_obj.dtype)
                if num_cls > 1:  # cls loss (only if multiple classes)
                    t = torch.zeros_like(ps[:, 5:], device=device)  # targets
                    t[range(n), targets_weights_cls[i][1]] = 1.
                    loss_cls += self.cls_bce(ps[:, 5:], t)  # BCE
            loss_obj += self.obj_bce(pi[..., 4], target_obj) * self.layer_balance[i]
        s = 3 / layer_num
        cls_weights = self.cls_weights * (num_cls / 80.)
        loss_box *= self.iou_weights * s
        loss_obj *= self.obj_weights * s
        loss_cls *= cls_weights * s
        loss = loss_box + loss_obj + loss_cls
        return loss * num_bs, torch.cat((loss_box, loss_obj, loss_cls, loss)).detach(), match_num
