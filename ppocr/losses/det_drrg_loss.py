# Copyright (c) OpenMMLab. All rights reserved.
import paddle
import paddle.nn.functional as F
from paddle import nn


class DRRGLoss(nn.Layer):
    """The class for implementing DRRG loss. This is partially adapted from
    https://github.com/GXYM/DRRG licensed under the MIT license.

    DRRG: `Deep Relational Reasoning Graph Network for Arbitrary Shape Text
    Detection <https://arxiv.org/abs/1908.05900>`_.

    Args:
        ohem_ratio (float): The negative/positive ratio in ohem.
    """

    def __init__(self, ohem_ratio=3.0):
        super().__init__()
        self.ohem_ratio = ohem_ratio
        self.downsample_ratio = 1.0

    def balance_bce_loss(self, pred, gt, mask):
        """Balanced Binary-CrossEntropy Loss.

        Args:
            pred (Tensor): Shape of :math:`(1, H, W)`.
            gt (Tensor): Shape of :math:`(1, H, W)`.
            mask (Tensor): Shape of :math:`(1, H, W)`.

        Returns:
            Tensor: Balanced bce loss.
        """
        assert pred.shape == gt.shape == mask.shape
        assert paddle.all(pred >= 0) and paddle.all(pred <= 1)
        assert paddle.all(gt >= 0) and paddle.all(gt <= 1)
        positive = gt * mask
        negative = (1 - gt) * mask
        positive_count = int(positive.sum())
        # gt = gt.float()
        if positive_count > 0:
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            positive_loss = paddle.sum(loss * positive)
            negative_loss = loss * negative
            negative_count = min(
                int(negative.sum()),
                int(positive_count * self.ohem_ratio))
        else:
            positive_loss = paddle.to_tensor(0.0)
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            negative_loss = loss * negative
            negative_count = 100
        negative_loss, _ = paddle.topk(negative_loss.reshape([-1]), negative_count)

        balance_loss = (positive_loss + paddle.sum(negative_loss)) / (
            float(positive_count + negative_count) + 1e-5)

        return balance_loss

    def gcn_loss(self, gcn_data):
        """CrossEntropy Loss from gcn module.

        Args:
            gcn_data (tuple(Tensor, Tensor)): The first is the
                prediction with shape :math:`(N, 2)` and the
                second is the gt label with shape :math:`(m, n)`
                where :math:`m * n = N`.

        Returns:
            Tensor: CrossEntropy loss.
        """
        gcn_pred, gt_labels = gcn_data
        gt_labels = gt_labels.reshape([-1])
        loss = F.cross_entropy(gcn_pred, gt_labels)

        return loss

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        # assert check_argument.is_type_list(bitmasks, BitmapMasks)
        # assert isinstance(target_sz, list)

        batch_size = len(bitmasks)
        # num_masks = len(bitmasks[0])

        results = []

        # for level_inx in range(num_masks):
        #     kernel = []
        kernel = []
        for batch_inx in range(batch_size):
            # mask = paddle.to_tensor(bitmasks[batch_inx].masks[level_inx])
            mask = bitmasks[batch_inx]
            # hxw
            mask_sz = mask.shape
            # left, right, top, bottom
            pad = [
                0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]
            ]
            mask = F.pad(mask, pad, mode='constant', value=0)
            kernel.append(mask)
        kernel = paddle.stack(kernel)
        results.append(kernel)

        return results

    def forward(self, preds, labels, ):
        """Compute Drrg loss.

        Args:
            preds (tuple(Tensor)): The first is the prediction map
                with shape :math:`(N, C_{out}, H, W)`.
                The second is prediction from GCN module, with
                shape :math:`(N, 2)`.
                The third is ground-truth label with shape :math:`(N, 8)`.
            downsample_ratio (float): The downsample ratio.
            gt_text_mask (list[BitmapMasks]): Text mask.
            gt_center_region_mask (list[BitmapMasks]): Center region mask.
            gt_mask (list[BitmapMasks]): Effective mask.
            gt_top_height_map (list[BitmapMasks]): Top height map.
            gt_bot_height_map (list[BitmapMasks]): Bottom height map.
            gt_sin_map (list[BitmapMasks]): Sinusoid map.
            gt_cos_map (list[BitmapMasks]): Cosine map.

        Returns:
            dict:  A loss dict with ``loss_text``, ``loss_center``,
            ``loss_height``, ``loss_sin``, ``loss_cos``, and ``loss_gcn``.
        """
        
        assert isinstance(preds, tuple)
        gt_text_mask, gt_center_region_mask, gt_mask, gt_top_height_map, gt_bot_height_map, gt_sin_map, gt_cos_map = labels[1:8]

        downsample_ratio = self.downsample_ratio

        pred_maps, gcn_data = preds
        pred_text_region = pred_maps[:, 0, :, :]
        pred_center_region = pred_maps[:, 1, :, :]
        pred_sin_map = pred_maps[:, 2, :, :]
        pred_cos_map = pred_maps[:, 3, :, :]
        pred_top_height_map = pred_maps[:, 4, :, :]
        pred_bot_height_map = pred_maps[:, 5, :, :]
        feature_sz = pred_maps.shape

        # bitmask 2 tensor
        mapping = {
            'gt_text_mask': paddle.cast(gt_text_mask, 'float32'),
            'gt_center_region_mask': paddle.cast(gt_center_region_mask, 'float32'),
            'gt_mask': paddle.cast(gt_mask, 'float32'),
            'gt_top_height_map': paddle.cast(gt_top_height_map, 'float32'),
            'gt_bot_height_map': paddle.cast(gt_bot_height_map, 'float32'),
            'gt_sin_map': paddle.cast(gt_sin_map, 'float32'),
            'gt_cos_map': paddle.cast(gt_cos_map, 'float32')
        }
        gt = {}
        for key, value in mapping.items():
            gt[key] = value
            if abs(downsample_ratio - 1.0) < 1e-2:
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            else:
                gt[key] = [item.rescale(downsample_ratio) for item in gt[key]]
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
                if key in ['gt_top_height_map', 'gt_bot_height_map']:
                    gt[key] = [item * downsample_ratio for item in gt[key]]
            gt[key] = [item for item in gt[key]]

        scale = paddle.sqrt(1.0 / (pred_sin_map**2 + pred_cos_map**2 + 1e-8))
        pred_sin_map = pred_sin_map * scale
        pred_cos_map = pred_cos_map * scale

        loss_text = self.balance_bce_loss(
            F.sigmoid(pred_text_region), gt['gt_text_mask'][0],
            gt['gt_mask'][0])

        text_mask = (gt['gt_text_mask'][0] * gt['gt_mask'][0])
        negative_text_mask = ((1 - gt['gt_text_mask'][0]) *
                              gt['gt_mask'][0])
        loss_center_map = F.binary_cross_entropy(
            F.sigmoid(pred_center_region),
            gt['gt_center_region_mask'][0],
            reduction='none')
        if int(text_mask.sum()) > 0:
            loss_center_positive = paddle.sum(
                loss_center_map * text_mask) / paddle.sum(text_mask)
        else:
            loss_center_positive = paddle.to_tensor(0.0)
        loss_center_negative = paddle.sum(
            loss_center_map *
            negative_text_mask) / paddle.sum(negative_text_mask)
        loss_center = loss_center_positive + 0.5 * loss_center_negative

        center_mask = (gt['gt_center_region_mask'][0] *
                       gt['gt_mask'][0])
        if int(center_mask.sum()) > 0:
            map_sz = pred_top_height_map.shape
            ones = paddle.ones(map_sz, dtype='float32')
            loss_top = F.smooth_l1_loss(
                pred_top_height_map / (gt['gt_top_height_map'][0] + 1e-2),
                ones,
                reduction='none')
            loss_bot = F.smooth_l1_loss(
                pred_bot_height_map / (gt['gt_bot_height_map'][0] + 1e-2),
                ones,
                reduction='none')
            gt_height = (
                gt['gt_top_height_map'][0] + gt['gt_bot_height_map'][0])
            loss_height = paddle.sum(
                (paddle.log(gt_height + 1) *
                 (loss_top + loss_bot)) * center_mask) / paddle.sum(center_mask)

            loss_sin = paddle.sum(
                F.smooth_l1_loss(
                    pred_sin_map, gt['gt_sin_map'][0], reduction='none') *
                center_mask) / paddle.sum(center_mask)
            loss_cos = paddle.sum(
                F.smooth_l1_loss(
                    pred_cos_map, gt['gt_cos_map'][0], reduction='none') *
                center_mask) / paddle.sum(center_mask)
        else:
            loss_height = paddle.to_tensor(0.0)
            loss_sin = paddle.to_tensor(0.0)
            loss_cos = paddle.to_tensor(0.0)

        loss_gcn = self.gcn_loss(gcn_data)

        loss = loss_text + loss_center + loss_height + loss_sin + loss_cos + loss_gcn
        results = dict(
            loss=loss,
            loss_text=loss_text,
            loss_center=loss_center,
            loss_height=loss_height,
            loss_sin=loss_sin,
            loss_cos=loss_cos,
            loss_gcn=loss_gcn)

        return results
