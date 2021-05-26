import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

from mmcv.cnn import ConvModule, bias_init_with_prob
import numpy as np
from mmdet.core import bbox_overlaps

INF = 1e8

@HEADS.register_module()
class FCOSHeadLGDet(AnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 kappa_scale=0.1,
                 delta_t=0.5,
                 max_steps=1,
                 less_convs=4,
                 box0=1.,
                 boxN=1.,
                 pretrain_mask=False,
                 train_all_box=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        
        self.pretrain_mask = pretrain_mask
        self.less_convs = less_convs
        self.box0 = box0
        self.boxN = boxN
        self.train_all_box = train_all_box
        self.sobel_kernel = torch.tensor([[[-1., 0, 1.]]], requires_grad=False).cuda()
        self.kappa_scale = kappa_scale
        self.delta_t = delta_t
        self.max_steps = max_steps
        soi = [rg[-1] for rg in self.regress_ranges]
        self.sizes_of_interest = torch.tensor(soi[:-1] + [soi[-2] * 2])
        self.mask_feat_channels = 8
        self.mask_out_stride = 4
        self.para_channels = 8
        self.num_layers = 3
        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                weight_nums.append((self.mask_feat_channels + 2) * self.para_channels)
                bias_nums.append(self.para_channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.para_channels*4)
                bias_nums.append(4)
            else:
                weight_nums.append(self.para_channels * self.para_channels)
                bias_nums.append(self.para_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        
    def _init_layers(self):
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        
        self.prm_convs = nn.ModuleList()#for parameters in CondInst
        self.mask_convs = nn.ModuleList()#for data and kappa
        for i in range(self.less_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.prm_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        self.prm_out = nn.Conv2d(self.feat_channels, self.num_gen_params, 3, padding=1)
        self.mask_out = nn.Conv2d(self.feat_channels, self.mask_feat_channels, 1)
        
    def init_weights(self):
        super().init_weights()
        normal_init(self.conv_centerness, std=0.01)
        
        for m in self.prm_convs:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        for m in self.mask_convs:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        normal_init(self.prm_out, std=0.01)
        normal_init(self.mask_out, std=0.01)
            
    def dice_loss(self, x, target):
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss
    
    def compute_locations(self, h, w, stride, device):
        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations
    
    def aligned_bilinear(self, tensor, factor):
        assert tensor.dim() == 4
        assert factor >= 1
        assert int(factor) == factor
        if factor == 1:
            return tensor
        h, w = tensor.size()[2:]
        tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
        oh = factor * h + 1
        ow = factor * w + 1
        tensor = F.interpolate(tensor, size=(oh, ow), mode='bilinear', align_corners=True)
        tensor = F.pad(tensor, pad=(factor // 2, 0, factor // 2, 0), mode="replicate")
        return tensor[:, :, :oh - 1, :ow - 1]
    
    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)
        
        num_insts = params.size(0)
        num_layers = len(weight_nums)
        
        params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))
        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]
        
        for l in range(num_layers):
            if l < num_layers - 1:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * 4, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts*4)
        return weight_splits, bias_splits

    def mask_heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x
    
    def mask_heads_forward_with_coords(self, mask_feats, mask_feat_stride, instances):
        locations = self.compute_locations(
                mask_feats.size(2), mask_feats.size(3),
                stride=mask_feat_stride, device=mask_feats.device)
        n_inst = len(instances['im_inds'])
        im_inds = instances['im_inds']
        mask_head_params = instances['mask_head_params']
        instance_locations = instances['locations']
        relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1).float()
        soi = self.sizes_of_interest.float()[instances['fpn_levels']].to(mask_feats.device)
        relative_coords = relative_coords / soi.reshape(-1, 1, 1)
        relative_coords = relative_coords.to(dtype=mask_feats.dtype)

        N, _, H, W = mask_feats.size()
        mask_head_inputs = torch.cat([relative_coords, mask_feats[im_inds].reshape(n_inst, 
                                      self.mask_feat_channels, H * W)], dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        weights, biases = self.parse_dynamic_params(mask_head_params, self.para_channels,
                                               self.weight_nums, self.bias_nums)
        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)
        mask_logits = mask_logits.reshape(-1, 4, H, W)
        
        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = self.aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))
        return mask_logits
    
    def target_para(self, target_bbox, height, width):
        coords_y = torch.arange(0, height*self.mask_out_stride, step=self.mask_out_stride
                                ).to(target_bbox.device)#no plus mask_out_stride/2, error will be larger
        coords_x = torch.arange(0, width*self.mask_out_stride, step=self.mask_out_stride
                                ).to(target_bbox.device)
        diff_xmin = target_bbox[:,0][...,None]-coords_x[None,:]
        diff_ymin = target_bbox[:,1][...,None]-coords_y[None,:]
        diff_xmax = target_bbox[:,2][...,None]-coords_x[None,:]
        diff_ymax = target_bbox[:,3][...,None]-coords_y[None,:]
        target_kappa_x = torch.min((-diff_xmin).clamp(min=0), 
                                   diff_xmax.clamp(min=0))
        target_kappa_y = torch.min((-diff_ymin).clamp(min=0),
                                   diff_ymax.clamp(min=0))
        
        d_data_dxmin = -F.conv1d(F.pad(abs(diff_xmin)[:,None], (1, 1), mode='replicate'), 
                                self.sobel_kernel)[:,0]
        d_data_dymin = -F.conv1d(F.pad(abs(diff_ymin)[:,None], (1, 1), mode='replicate'), 
                                self.sobel_kernel)[:,0]
        d_data_dxmax = F.conv1d(F.pad(abs(diff_xmax)[:,None], (1, 1), mode='replicate'), 
                                self.sobel_kernel)[:,0]
        d_data_dymax = F.conv1d(F.pad(abs(diff_ymax)[:,None], (1, 1), mode='replicate'), 
                                self.sobel_kernel)[:,0]
        target_data_x = torch.max(d_data_dxmin, d_data_dxmax)
        target_data_y = torch.max(d_data_dymin, d_data_dymax)
        return target_data_x, target_data_y, target_kappa_x, target_kappa_y
    
    def forward(self, feats):
        cls_scores, bbox_preds, centernesses, parameters = multi_apply(
                self.forward_single, feats, self.scales, self.strides)
        return cls_scores, bbox_preds, centernesses, parameters, feats[0]

    def forward_single(self, x, scale, stride):
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
            
        prm_feat = x
        for prm_layer in self.prm_convs:
            prm_feat = prm_layer(prm_feat)
        parameters = self.prm_out(prm_feat)
        return cls_score, bbox_pred, centerness, parameters

    def evolve_active_rays(self, stride, rhos, bboxes0, data_x, data_y, kappa_x, kappa_y):
        bboxes = bboxes0/self.mask_out_stride
        height = data_y.size(-1)
        width = data_x.size(-1)
        kappa_x = kappa_x*self.kappa_scale
        kappa_y = kappa_y*self.kappa_scale
        coords_norm_x = (bboxes[:,[0,2]]-(width-1)/2)/(width-1)*2
        coords_norm_y = (bboxes[:,[1,3]]-(height-1)/2)/(height-1)*2
        coords_norm = torch.cat([coords_norm_x, coords_norm_y], -1)#(xmin,xmax,ymin,ymax)
        coords_sample = torch.stack([coords_norm.new_zeros(coords_norm.size()),
                                     coords_norm], -1)[:,None]#(num_point, 1, 4, 2)

        d_data_dxmin_i = F.grid_sample(data_x[:,None,:,None], coords_sample[:,:,:1], 
                                       padding_mode="border")[:,0,0]#(num_point, 1)
        d_data_dxmax_i = F.grid_sample(data_x[:,None,:,None], coords_sample[:,:,1:2], 
                                       padding_mode="border")[:,0,0]#(num_point, 1)
        d_data_dymin_i = F.grid_sample(data_y[:,None,:,None], coords_sample[:,:,2:3], 
                                       padding_mode="border")[:,0,0]#(num_point, 1)
        d_data_dymax_i = F.grid_sample(data_y[:,None,:,None], coords_sample[:,:,3:], 
                                       padding_mode="border")[:,0,0]#(num_point, 1)
        d_data_dxmin_i[coords_sample[:,0,0,1]>1] = 0
        d_data_dxmin_i[coords_sample[:,0,0,1]<-1] = 0
        d_data_dxmax_i[coords_sample[:,0,1,1]>1] = 0
        d_data_dxmax_i[coords_sample[:,0,1,1]<-1] = 0
        d_data_dymin_i[coords_sample[:,0,2,1]>1] = 0
        d_data_dymin_i[coords_sample[:,0,2,1]<-1] = 0
        d_data_dymax_i[coords_sample[:,0,3,1]>1] = 0
        d_data_dymax_i[coords_sample[:,0,3,1]<-1] = 0
        
        kappa_i_xmin = F.grid_sample(kappa_x[:,None,:,None], 
                                     coords_sample[:,:,:1],
                                     padding_mode="border")[:,0,0]#(num_point, 1)
        kappa_i_xmax = F.grid_sample(kappa_x[:,None,:,None], 
                                     coords_sample[:,:,1:2],
                                     padding_mode="border")[:,0,0]#(num_point, 1)
        kappa_i_ymin = F.grid_sample(kappa_y[:,None,:,None], 
                                     coords_sample[:,:,2:3],
                                     padding_mode="border")[:,0,0]#(num_point, 1)
        kappa_i_ymax = F.grid_sample(kappa_y[:,None,:,None],
                                     coords_sample[:,:,3:],
                                     padding_mode="border")[:,0,0]#(num_point, 1)
        kappa_i_xmin[coords_sample[:,0,0,1]>1] = 0
        kappa_i_xmin[coords_sample[:,0,0,1]<-1] = 0
        kappa_i_xmax[coords_sample[:,0,1,1]>1] = 0
        kappa_i_xmax[coords_sample[:,0,1,1]<-1] = 0
        kappa_i_ymin[coords_sample[:,0,2,1]>1] = 0
        kappa_i_ymin[coords_sample[:,0,2,1]<-1] = 0
        kappa_i_ymax[coords_sample[:,0,3,1]>1] = 0
        kappa_i_ymax[coords_sample[:,0,3,1]<-1] = 0
        
        update = torch.cat([d_data_dxmin_i-kappa_i_xmin, d_data_dymin_i-kappa_i_ymin,
                            d_data_dxmax_i-kappa_i_xmax, d_data_dymax_i-kappa_i_ymax], -1)
        rhos = (rhos*stride-self.delta_t*update).clamp(min=0)
        return rhos/stride
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores, bbox_preds, centernesses, parameters, feat0,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        
        flatten_prm = torch.cat([prm.permute(0, 2, 3, 1).reshape(-1, self.num_gen_params)
                                  for prm in parameters])
        level_ids = torch.cat([(i*points.new_ones(points.size(0))).long().repeat(num_imgs)
                               for i, points in enumerate(all_level_points)])
        img_ids0 = torch.arange(num_imgs).to(level_ids.device)
        img_ids = torch.cat([img_ids0[:,None].repeat(1, points.size(0)).reshape(-1)
                             for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        losses = {}
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)
        losses['loss_cls'] = loss_cls
        
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if len(pos_inds) > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            centerness_denorm = max(
                reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
            losses['loss_bbox0'] = loss_bbox*self.box0
            losses['loss_centerness'] = loss_centerness
            
            strides = torch.tensor(self.strides)[level_ids].to(pos_inds.device).reshape(-1, 1)
            if self.norm_on_bbox:
                gt_targets = strides[pos_inds]*pos_bbox_targets
                gt_target_decoded = distance2bbox(pos_points, gt_targets)
            else:
                gt_targets = pos_bbox_targets
                gt_target_decoded = pos_decoded_target_preds            

            instances = {}
            instances['im_inds'] = img_ids[pos_inds]
            instances['fpn_levels'] = level_ids[pos_inds]
            instances['mask_head_params'] = flatten_prm[pos_inds]
            instances['locations'] = flatten_points[pos_inds]
            mask_feat = feat0
            for mask_layer in self.mask_convs:
                mask_feat = mask_layer(mask_feat)
            mask = self.mask_out(mask_feat)
            mask_logits = self.mask_heads_forward_with_coords(
                    mask, self.strides[0], instances)#(data_y, data_x, kappa_y, kappa_x)
            data_y = mask_logits[:,0].max(2)[0]
            data_x = mask_logits[:,1].max(1)[0]
            kappa_y = F.relu(mask_logits[:,2].max(2)[0])
            kappa_x = F.relu(mask_logits[:,3].max(1)[0])
            new_h = data_y.size(-1)
            new_w = data_x.size(-1)
            target_data_x, target_data_y, target_kappa_x, target_kappa_y = self.target_para(
                    gt_target_decoded, new_h, new_w)
            loss_data_x = self.dice_loss(data_x, target_data_x).mean()
            loss_data_y = self.dice_loss(data_y, target_data_y).mean()
            loss_kappa_x = self.dice_loss(kappa_x, target_kappa_x).mean()
            loss_kappa_y = self.dice_loss(kappa_y, target_kappa_y).mean()
            loss_data = loss_data_x+loss_data_y
            loss_kappa = loss_kappa_x+loss_kappa_y
            losses['loss_data'] = loss_data
            if self.kappa_scale:
                losses['loss_kappa'] = loss_kappa
            if not self.pretrain_mask:
                pos_rho = pos_bbox_preds
                aux_preds = distance2bbox(pos_points, strides[pos_inds]*pos_rho)
                for i in range(self.max_steps):
                    pos_rho = self.evolve_active_rays(
                            strides[pos_inds],
                            pos_rho, aux_preds, data_x, data_y, kappa_x, kappa_y)
                    aux_preds = distance2bbox(pos_points, strides[pos_inds]*pos_rho)
                    loss_bbox = self.loss_bbox(
                            distance2bbox(pos_points, pos_rho),
                            pos_decoded_target_preds,
                            weight=pos_centerness_targets,
                            avg_factor=centerness_denorm)
                    if self.train_all_box:
                        losses['loss_bbox{}'.format(i+1)] = loss_bbox/self.max_steps
                    else:
                        if i==self.max_steps-1:
                            losses['loss_bbox{}'.format(i+1)] = loss_bbox*self.boxN
        else:
            if self.pretrain_mask:
                loss_data = pos_bbox_preds.sum()
                loss_kappa = pos_bbox_preds.sum()            
            else:
                losses['loss_bbox0'] = pos_bbox_preds.sum()
                for i in range(self.max_steps):
                    if self.train_all_box:
                        losses['loss_bbox{}'.format(i+1)] = pos_bbox_preds.sum()
                    else:
                        if i==self.max_steps-1:
                            losses['loss_bbox{}'.format(i+1)] = pos_bbox_preds.sum()
                losses['loss_data'] = pos_bbox_preds.sum()
                if self.kappa_scale:
                    losses['loss_kappa'] = pos_bbox_preds.sum()
                losses['loss_centerness'] = pos_centerness.sum()
        if self.pretrain_mask:
            return dict(
                    loss_data=loss_data,
                    loss_kappa=loss_kappa)
        else:
            return losses

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores, bbox_preds, centernesses, parameters, feat0,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]
        parameter_list = [parameters[i].detach() for i in range(num_levels)]
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]        
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, 
                                       parameter_list, feat0,
                                       mlvl_points,
                                       img_shapes, scale_factors, cfg, rescale,
                                       with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores, bbox_preds, centernesses,
                    parameter, feat0,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_param = []
        mlvl_level_ids = []
        mlvl_img_ids = []
        mlvl_point = []
        mlvl_bbox_preds = []
        level_id = 0
        for cls_score, bbox_pred, centerness, points, param in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points, parameter):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            param = param.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_gen_params)
            level_ids = (level_id*points.new_ones(centerness.size())).long()
            img_ids = torch.arange(batch_size).view(-1, 1).expand_as(level_ids).long()
            # Always keep topk op for dynamic input in onnx
            if nms_pre_tensor > 0 and (torch.onnx.is_in_onnx_export()
                                       or scores.shape[-2] > nms_pre_tensor):
                from torch import _shape_as_tensor
                # keep shape as tensor and get k
                num_anchor = _shape_as_tensor(scores)[-2].to(device)
                nms_pre = torch.where(nms_pre_tensor < num_anchor,
                                      nms_pre_tensor, num_anchor)
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
                centerness = centerness[batch_inds, topk_inds]
                param = param[batch_inds, topk_inds, :]
                level_ids = level_ids[batch_inds, topk_inds]
                img_ids = img_ids[batch_inds, topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_param.append(param)
            mlvl_level_ids.append(level_ids)
            mlvl_img_ids.append(img_ids)
            mlvl_point.append(points.reshape(batch_size, -1, 2))
            mlvl_bbox_preds.append(bbox_pred)
            
            level_id += 1      

        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
        batch_mlvl_param = torch.cat(mlvl_param, dim=1)
        batch_mlvl_level_ids = torch.cat(mlvl_level_ids, dim=1)
        batch_mlvl_img_ids = torch.cat(mlvl_img_ids, dim=1)
        batch_mlvl_point = torch.cat(mlvl_point, dim=1)
        batch_mlvl_bbox_preds = torch.cat(mlvl_bbox_preds, dim=1)
        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        
        #######################################################################
        if rescale:
            batch_mlvl_bboxes_rescale = batch_mlvl_bboxes/batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        else:
            batch_mlvl_bboxes_rescale = batch_mlvl_bboxes
        deploy_nms_pre = cfg.get('deploy_nms_pre', -1)
        if deploy_nms_pre > 0 and torch.onnx.is_in_onnx_export():
            batch_mlvl_scores, _ = (
                batch_mlvl_scores *
                batch_mlvl_centerness.unsqueeze(2).expand_as(batch_mlvl_scores)
            ).max(-1)
            _, topk_inds = batch_mlvl_scores.topk(deploy_nms_pre)
            batch_inds = torch.arange(batch_mlvl_scores.shape[0]).view(
                -1, 1).expand_as(topk_inds)
            batch_mlvl_scores = batch_mlvl_scores[batch_inds, topk_inds, :]
            batch_mlvl_bboxes_rescale = batch_mlvl_bboxes_rescale[batch_inds, topk_inds, :]
            batch_mlvl_bboxes = batch_mlvl_bboxes[batch_inds, topk_inds, :]
            batch_mlvl_centerness = batch_mlvl_centerness[batch_inds,
                                                          topk_inds]
            batch_mlvl_param = batch_mlvl_param[batch_inds, topk_inds]
            batch_mlvl_level_ids = batch_mlvl_level_ids[batch_inds, topk_inds]
            batch_mlvl_img_ids = batch_mlvl_img_ids[batch_inds, topk_inds]
            batch_mlvl_point = batch_mlvl_point[batch_inds, topk_inds]
            batch_mlvl_bbox_preds = batch_mlvl_bbox_preds[batch_inds, topk_inds]
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)
        mask_feat = feat0
        for mask_layer in self.mask_convs:
            mask_feat = mask_layer(mask_feat)
        mask = self.mask_out(mask_feat)
        if with_nms:
            det_results = []
            for (mlvl_bboxes_rescale, mlvl_scores, mlvl_centerness, mlvl_param, mlvl_level_id,
                 mlvl_img_id, mlvl_point, mlvl_bbox_preds, mlvl_bboxes, scale_factor) in zip(
                    batch_mlvl_bboxes_rescale, batch_mlvl_scores, batch_mlvl_centerness,
                    batch_mlvl_param, batch_mlvl_level_ids, batch_mlvl_img_ids,
                    batch_mlvl_point, batch_mlvl_bbox_preds, batch_mlvl_bboxes, scale_factors):
                det_bbox, det_label, keep = multiclass_nms(
                    mlvl_bboxes_rescale,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness,
                    return_inds=True)
                if len(keep):
                    if mlvl_bboxes.shape[1] > 4:
                        bboxes = mlvl_bboxes.view(mlvl_scores.size(0), -1, 4)
                        bboxes_preds = mlvl_bbox_preds.view(mlvl_scores.size(0), -1, 4)
                    else:
                        bboxes = mlvl_bboxes[:, None].expand(
                            mlvl_scores.size(0), self.num_classes, 4)
                        bboxes_preds = mlvl_bbox_preds[:, None].expand(
                            mlvl_scores.size(0), self.num_classes, 4)
                    params = mlvl_param[:, None].expand(
                        mlvl_scores.size(0), self.num_classes, self.num_gen_params)
                    points = mlvl_point[:, None].expand(
                        mlvl_scores.size(0), self.num_classes, 2)
                    scores = mlvl_scores[:, :-1]
                    mlvl_level_id = mlvl_level_id[:, None].expand_as(scores)
                    mlvl_img_id = mlvl_img_id[:, None].expand_as(scores)
                    bboxes = bboxes.reshape(-1, 4)
                    points = points.reshape(-1, 2)
                    params = params.reshape(-1, self.num_gen_params)
                    bboxes_preds = bboxes_preds.reshape(-1, 4)
                    scores = scores.reshape(-1)
                    mlvl_level_id = mlvl_level_id.reshape(-1)
                    mlvl_img_id = mlvl_img_id.reshape(-1)
                    if not torch.onnx.is_in_onnx_export():
                        valid_mask = scores > cfg.score_thr
                    score_factors = mlvl_centerness.view(-1, 1).expand(
                        mlvl_scores.size(0), self.num_classes)
                    score_factors = score_factors.reshape(-1)
                    scores = scores * score_factors
                    if not torch.onnx.is_in_onnx_export():
                        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
                        bboxes, bboxes_preds = bboxes[inds], bboxes_preds[inds]
                        points, params, scores = points[inds], params[inds], scores[inds]
                        mlvl_level_id, mlvl_img_id = mlvl_level_id[inds], mlvl_img_id[inds]
                    else:
                        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)])
                        points = torch.cat([points, points.new_zeros(1, 2)])
                        scores = torch.cat([scores, scores.new_zeros(1)])
                        mlvl_img_id = torch.cat([mlvl_img_id, mlvl_img_id.new_zeros(1)])
                        mlvl_level_id = torch.cat([mlvl_level_id, mlvl_level_id.new_zeros(1)])
                        params = torch.cat([params, params.new_zeros(1, self.num_gen_params)])
                        bboxes_preds = torch.cat([bboxes_preds, bboxes_preds.new_zeros(1, 4)])
                    points = points[keep]
                    bboxes = bboxes[keep]
                    scores = scores[keep]
                    mlvl_img_id = mlvl_img_id[keep]
                    mlvl_level_id = mlvl_level_id[keep]
                    params = params[keep]
                    bboxes_preds = bboxes_preds[keep]
                    instances = {}
                    instances['im_inds'] = mlvl_img_id.reshape(-1)
                    instances['fpn_levels'] = mlvl_level_id.reshape(-1)
                    instances['mask_head_params'] = params.reshape(-1, self.num_gen_params)
                    instances['locations'] = points.reshape(-1, 2)
                    mask_logits = self.mask_heads_forward_with_coords(
                            mask, self.strides[0], instances)#(data_y, data_x, kappa_y, kappa_x)
                    data_y = mask_logits[:,0].max(2)[0]
                    data_x = mask_logits[:,1].max(1)[0]
                    kappa_y = F.relu(mask_logits[:,2].max(2)[0])
                    kappa_x = F.relu(mask_logits[:,3].max(1)[0])
                    pos_rho = bboxes_preds.reshape(-1, 4)
                    aux_preds = bboxes.reshape(-1, 4)
                    for i in range(self.max_steps):
                        pos_rho = self.evolve_active_rays(pos_rho.new_ones(pos_rho.size(0), 1),
                                pos_rho, aux_preds, data_x, data_y, kappa_x, kappa_y)
                        aux_preds = distance2bbox(points.reshape(-1, 2), pos_rho)
                    bboxes = aux_preds
                    if rescale:
                        det_bbox = bboxes/bboxes.new_tensor(scale_factor)
                    else:
                        det_bbox = bboxes
                    det_bbox = torch.cat([det_bbox, scores[...,None]], 1)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)