import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init
from mmcv.ops import DeformConv2d

from mmdet.core import multi_apply, multiclass_nms
from ..builder import HEADS
from .anchor_free_head import AnchorFreeHead

import numpy as np
import torch.nn.functional as F

INF = 1e8

class FeatureAlign(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deform_groups=4):
        super(FeatureAlign, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            4, deform_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape)
        x = self.relu(self.conv_adaption(x, offset))
        return x

@HEADS.register_module()
class FoveaHeadLGDet(AnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128,
                                                                         512)),
                 sigma=0.4,
                 with_deform=False,
                 pretrain_mask=False,
                 deform_groups=4,
                 max_steps=2,
                 **kwargs):
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        self.with_deform = with_deform
        self.deform_groups = deform_groups
        
        self.pretrain_mask = pretrain_mask
        self.sobel_kernel = torch.tensor([[[-1., 0, 1.]]], requires_grad=False).cuda()
        self.kappa_scale = 0.1
        self.delta_t = 0.5
        self.max_steps = max_steps
        soi = [rg[-1] for rg in self.scale_ranges]
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
        
        super().__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self):
        # box branch
        super()._init_reg_convs()
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        
        self.prm_convs = nn.ModuleList()#for parameters in CondInst
        self.mask_convs = nn.ModuleList()#for data and kappa
        for i in range(self.stacked_convs):
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
        
        # cls branch
        if not self.with_deform:
            super()._init_cls_convs()
            self.conv_cls = nn.Conv2d(
                self.feat_channels, self.cls_out_channels, 3, padding=1)
        else:
            self.cls_convs = nn.ModuleList()
            self.cls_convs.append(
                ConvModule(
                    self.feat_channels, (self.feat_channels * 4),
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.cls_convs.append(
                ConvModule((self.feat_channels * 4), (self.feat_channels * 4),
                           1,
                           stride=1,
                           padding=0,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg,
                           bias=self.norm_cfg is None))
            self.feature_adaption = FeatureAlign(
                self.feat_channels,
                self.feat_channels,
                kernel_size=3,
                deform_groups=self.deform_groups)
            self.conv_cls = nn.Conv2d(
                int(self.feat_channels * 4),
                self.cls_out_channels,
                3,
                padding=1)

    def init_weights(self):
        super().init_weights()
        if self.with_deform:
            self.feature_adaption.init_weights()
            
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

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        if self.with_deform:
            cls_feat = self.feature_adaption(cls_feat, bbox_pred.exp())
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        
        prm_feat = x
        for prm_layer in self.prm_convs:
            prm_feat = prm_layer(prm_feat)
        parameters = self.prm_out(prm_feat)
        return cls_score, bbox_pred, parameters
    
    def forward(self, feats):
        cls_scores, bbox_preds, parameters = multi_apply(
                self.forward_single, feats)
        return cls_scores, bbox_preds, parameters, feats[0]

    def _get_points_single(self, *args, **kwargs):
        y, x = super()._get_points_single(*args, **kwargs)
        return y + 0.5, x + 0.5
    
    def evolve_active_rays(self, stride, rhos, bboxes0, data_x, data_y, kappa_x, kappa_y):
        bboxes = bboxes0/self.mask_out_stride
        height = data_y.size(-1)
        width = data_x.size(-1)
        kappa_x = kappa_x*self.kappa_scale
        kappa_y = kappa_y*self.kappa_scale
        coords_norm_x = (bboxes[:,[0,2]]-(width-1)/2)/(width-1)*2
        coords_norm_y = (bboxes[:,[1,3]]-(height-1)/2)/(height-1)*2
        coords_norm = torch.cat([coords_norm_x, coords_norm_y], -1)#(xmin,xmax,ymin,ymax)
        coords_norm = coords_norm.clamp(max=1)
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
        rhos = (rhos*stride-self.delta_t*update).clamp(min=1e-4)
        return rhos/stride
    
    def loss(self,
             cls_scores,
             bbox_preds,
             parameters, feat0,
             gt_bbox_list,
             gt_label_list,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                 bbox_preds[0].device)
        all_level_points = [torch.stack([point[1].reshape(-1)*self.strides[lv_i], 
                                         point[0].reshape(-1)*self.strides[lv_i]],-1) 
                            for lv_i, point in enumerate(points)]
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_labels, flatten_bbox_targets = self.get_targets(
            gt_bbox_list, gt_label_list, featmap_sizes, points)
        
        flatten_prm = torch.cat([prm.permute(0, 2, 3, 1).reshape(-1, self.num_gen_params)
                                  for prm in parameters])
        level_ids = torch.cat([(i*point.new_ones(point.size(0))).long().repeat(num_imgs)
                               for i, point in enumerate(all_level_points)])
        img_ids0 = torch.arange(num_imgs).to(level_ids.device)
        img_ids = torch.cat([img_ids0[:,None].repeat(1, point.size(0)).reshape(-1)
                             for point in all_level_points])
        flatten_points = torch.cat(
            [point.repeat(num_imgs, 1) for point in all_level_points])
        
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < self.num_classes)).nonzero().view(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos + num_imgs)
        if num_pos > 0:
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_weights = pos_bbox_targets.new_zeros(
                pos_bbox_targets.size()) + 1.0
            loss_bbox = self.loss_bbox(
                pos_bbox_preds,
                pos_bbox_targets,
                pos_weights,
                avg_factor=num_pos)
            
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
            base_len = torch.tensor(self.base_edge_list).to(level_ids.device)[level_ids][pos_inds]
            x1 = flatten_points[pos_inds,0] - base_len * pos_bbox_targets.exp()[:, 0]
            y1 = flatten_points[pos_inds,1] - base_len * pos_bbox_targets.exp()[:, 1]
            x2 = flatten_points[pos_inds,0] + base_len * pos_bbox_targets.exp()[:, 2]
            y2 = flatten_points[pos_inds,1] + base_len * pos_bbox_targets.exp()[:, 3]
            gt_target_decoded = torch.stack([x1, y1, x2, y2], -1)
            target_data_x, target_data_y, target_kappa_x, target_kappa_y = self.target_para(
                    gt_target_decoded, new_h, new_w)
            loss_data_x = self.dice_loss(data_x, target_data_x).mean()
            loss_data_y = self.dice_loss(data_y, target_data_y).mean()
            loss_kappa_x = self.dice_loss(kappa_x, target_kappa_x).mean()
            loss_kappa_y = self.dice_loss(kappa_y, target_kappa_y).mean()
            loss_data = loss_data_x+loss_data_y
            loss_kappa = loss_kappa_x+loss_kappa_y

            img_h, img_w = featmap_sizes[0]
            img_h, img_w = img_h*self.strides[0], img_w*self.strides[0]
            pos_rho = pos_bbox_preds.exp()
            x1 = (flatten_points[pos_inds,0] - base_len * pos_rho[:, 0]
            ).clamp(min=0, max=img_w-1)
            y1 = (flatten_points[pos_inds,1] - base_len * pos_rho[:, 1]
            ).clamp(min=0, max=img_h-1)
            x2 = (flatten_points[pos_inds,0] + base_len * pos_rho[:, 2]
            ).clamp(min=0, max=img_w-1)
            y2 = (flatten_points[pos_inds,1] + base_len * pos_rho[:, 3]
            ).clamp(min=0, max=img_h-1)
            aux_preds = torch.stack([x1, y1, x2, y2], -1)
            for i in range(self.max_steps):
                pos_rho = self.evolve_active_rays(
                        base_len[...,None],
                        pos_rho, aux_preds, data_x, data_y, kappa_x, kappa_y)
                x1 = (flatten_points[pos_inds,0] - base_len * pos_rho[:, 0]
                ).clamp(min=0, max=img_w-1)
                y1 = (flatten_points[pos_inds,1] - base_len * pos_rho[:, 1]
                ).clamp(min=0, max=img_h-1)
                x2 = (flatten_points[pos_inds,0] + base_len * pos_rho[:, 2]
                ).clamp(min=0, max=img_w-1)
                y2 = (flatten_points[pos_inds,1] + base_len * pos_rho[:, 3]
                ).clamp(min=0, max=img_h-1)
                aux_preds = torch.stack([x1, y1, x2, y2], -1)
                if i==self.max_steps-1:
                    loss_bboxN = self.loss_bbox(
                                        torch.log(pos_rho),
                                        pos_bbox_targets,
                                        pos_weights,
                                        avg_factor=num_pos)                        
        else:
            loss_bbox = torch.tensor(
                0,
                dtype=flatten_bbox_preds.dtype,
                device=flatten_bbox_preds.device)
            loss_bboxN = torch.tensor(
                0,
                dtype=flatten_bbox_preds.dtype,
                device=flatten_bbox_preds.device)
            loss_data = torch.tensor(
                0,
                dtype=flatten_bbox_preds.dtype,
                device=flatten_bbox_preds.device)
            loss_kappa= torch.tensor(
                0,
                dtype=flatten_bbox_preds.dtype,
                device=flatten_bbox_preds.device)
        if self.pretrain_mask:
            return dict(loss_data=loss_data, loss_kappa=loss_kappa)
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_bboxN=loss_bboxN,
                    loss_data=loss_data, loss_kappa=loss_kappa)

    def get_targets(self, gt_bbox_list, gt_label_list, featmap_sizes, points):
        label_list, bbox_target_list = multi_apply(
            self._get_target_single,
            gt_bbox_list,
            gt_label_list,
            featmap_size_list=featmap_sizes,
            point_list=points)
        flatten_labels = [
            torch.cat([
                labels_level_img.flatten() for labels_level_img in labels_level
            ]) for labels_level in zip(*label_list)
        ]
        flatten_bbox_targets = [
            torch.cat([
                bbox_targets_level_img.reshape(-1, 4)
                for bbox_targets_level_img in bbox_targets_level
            ]) for bbox_targets_level in zip(*bbox_target_list)
        ]
        flatten_labels = torch.cat(flatten_labels)
        flatten_bbox_targets = torch.cat(flatten_bbox_targets)
        return flatten_labels, flatten_bbox_targets

    def _get_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           featmap_size_list=None,
                           point_list=None):

        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) *
                              (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        label_list = []
        bbox_target_list = []
        # for each pyramid, find the cls and box target
        for base_len, (lower_bound, upper_bound), stride, featmap_size, \
            (y, x) in zip(self.base_edge_list, self.scale_ranges,
                          self.strides, featmap_size_list, point_list):
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            labels = gt_labels_raw.new_zeros(featmap_size) + self.num_classes
            bbox_targets = gt_bboxes_raw.new(featmap_size[0], featmap_size[1],
                                             4) + 1
            # scale assignment
            hit_indices = ((gt_areas >= lower_bound) &
                           (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                label_list.append(labels)
                bbox_target_list.append(torch.log(bbox_targets))
                continue
            _, hit_index_order = torch.sort(-gt_areas[hit_indices])
            hit_indices = hit_indices[hit_index_order]
            gt_bboxes = gt_bboxes_raw[hit_indices, :] / stride
            gt_labels = gt_labels_raw[hit_indices]
            half_w = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0])
            half_h = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            # valid fovea area: left, right, top, down
            pos_left = torch.ceil(
                gt_bboxes[:, 0] + (1 - self.sigma) * half_w - 0.5).long().\
                clamp(0, featmap_size[1] - 1)
            pos_right = torch.floor(
                gt_bboxes[:, 0] + (1 + self.sigma) * half_w - 0.5).long().\
                clamp(0, featmap_size[1] - 1)
            pos_top = torch.ceil(
                gt_bboxes[:, 1] + (1 - self.sigma) * half_h - 0.5).long().\
                clamp(0, featmap_size[0] - 1)
            pos_down = torch.floor(
                gt_bboxes[:, 1] + (1 + self.sigma) * half_h - 0.5).long().\
                clamp(0, featmap_size[0] - 1)
            for px1, py1, px2, py2, label, (gt_x1, gt_y1, gt_x2, gt_y2) in \
                    zip(pos_left, pos_top, pos_right, pos_down, gt_labels,
                        gt_bboxes_raw[hit_indices, :]):
                labels[py1:py2 + 1, px1:px2 + 1] = label
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 0] = \
                    (stride * x[py1:py2 + 1, px1:px2 + 1] - gt_x1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 1] = \
                    (stride * y[py1:py2 + 1, px1:px2 + 1] - gt_y1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 2] = \
                    (gt_x2 - stride * x[py1:py2 + 1, px1:px2 + 1]) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 3] = \
                    (gt_y2 - stride * y[py1:py2 + 1, px1:px2 + 1]) / base_len
            bbox_targets = bbox_targets.clamp(min=1. / 16, max=16.)
            label_list.append(labels)
            bbox_target_list.append(torch.log(bbox_targets))
        return label_list, bbox_target_list

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   parameters, feat0,
                   img_metas,
                   cfg=None,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = self.get_points(
            featmap_sizes,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
            flatten=True)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            parameter_list = [
                parameters[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list, bbox_pred_list, 
                                                 parameter_list, feat0, img_id,
                                                 featmap_sizes,
                                                 points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           parameters, feat0, img_id,
                           featmap_sizes,
                           point_list,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(point_list)
        det_bboxes = []
        det_scores = []
        mlvl_param = []
        mlvl_level_ids = []
        mlvl_img_ids = []
        mlvl_point = []
        mlvl_bbox_preds = []
        mlvl_base_len = []
        level_id = 0
        img_h, img_w = img_shape[0], img_shape[1]
        for cls_score, bbox_pred, stride, base_len, (y, x), param \
                in zip(cls_scores, bbox_preds, self.strides, self.base_edge_list, 
                       point_list, parameters):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4).exp()
            param = param.permute(1, 2, 0).reshape(-1, self.num_gen_params)
            level_ids = (level_id*param.new_ones(param.size(0))).long()
            img_ids = img_id*param.new_ones(param.size(0)).long()
            nms_pre = cfg.get('nms_pre', -1)
            if (nms_pre > 0) and (scores.shape[0] > nms_pre):
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                param = param[topk_inds, :]
                level_ids = level_ids[topk_inds]
                img_ids = img_ids[topk_inds]
                y = y[topk_inds]
                x = x[topk_inds]
            point = torch.stack([x*stride, y*stride],-1) 
            x1 = (stride * x - base_len * bbox_pred[:, 0]).\
                clamp(min=0, max=img_shape[1] - 1)
            y1 = (stride * y - base_len * bbox_pred[:, 1]).\
                clamp(min=0, max=img_shape[0] - 1)
            x2 = (stride * x + base_len * bbox_pred[:, 2]).\
                clamp(min=0, max=img_shape[1] - 1)
            y2 = (stride * y + base_len * bbox_pred[:, 3]).\
                clamp(min=0, max=img_shape[0] - 1)
            bboxes = torch.stack([x1, y1, x2, y2], -1)
            det_bboxes.append(bboxes)
            det_scores.append(scores)
            mlvl_param.append(param)
            mlvl_level_ids.append(level_ids)
            mlvl_img_ids.append(img_ids)
            mlvl_point.append(point)
            mlvl_bbox_preds.append(bbox_pred)
            mlvl_base_len.append(base_len*bbox_pred.new_ones(bbox_pred.size(0)))
            level_id += 1
        det_bboxes = torch.cat(det_bboxes)
        det_scores = torch.cat(det_scores)
        mlvl_param = torch.cat(mlvl_param)
        mlvl_level_ids = torch.cat(mlvl_level_ids)
        mlvl_img_ids = torch.cat(mlvl_img_ids)
        mlvl_point = torch.cat(mlvl_point)
        mlvl_bbox_preds = torch.cat(mlvl_bbox_preds)
        mlvl_base_len = torch.cat(mlvl_base_len)
        
        max_scores, _ = det_scores.max(-1)
        _, topk_inds = max_scores.topk(1000)
        det_scores = det_scores[topk_inds, :]
        det_bboxes = det_bboxes[topk_inds, :]
        mlvl_param = mlvl_param[topk_inds, :]
        mlvl_level_ids = mlvl_level_ids[topk_inds]
        mlvl_img_ids = mlvl_img_ids[topk_inds]
        mlvl_point = mlvl_point[topk_inds, :]
        mlvl_bbox_preds = mlvl_bbox_preds[topk_inds, :]
        mlvl_base_len = mlvl_base_len[topk_inds]
        instances = {}
        instances['im_inds'] = mlvl_img_ids
        instances['fpn_levels'] = mlvl_level_ids
        instances['mask_head_params'] = mlvl_param
        instances['locations'] = mlvl_point
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
        pos_rho = mlvl_bbox_preds
        x1 = (mlvl_point[:,0] - mlvl_base_len * pos_rho[:, 0]
        ).clamp(min=0, max=img_w-1)
        y1 = (mlvl_point[:,1] - mlvl_base_len * pos_rho[:, 1]
        ).clamp(min=0, max=img_h-1)
        x2 = (mlvl_point[:,0] + mlvl_base_len * pos_rho[:, 2]
        ).clamp(min=0, max=img_w-1)
        y2 = (mlvl_point[:,1] + mlvl_base_len * pos_rho[:, 3]
        ).clamp(min=0, max=img_h-1)
        aux_preds = torch.stack([x1, y1, x2, y2], -1)
        for i in range(self.max_steps):
            pos_rho = self.evolve_active_rays(
                    mlvl_base_len[...,None],
                    pos_rho, aux_preds, data_x, data_y, kappa_x, kappa_y)
            x1 = (mlvl_point[:,0] - mlvl_base_len * pos_rho[:, 0]
            ).clamp(min=0, max=img_w-1)
            y1 = (mlvl_point[:,1] - mlvl_base_len * pos_rho[:, 1]
            ).clamp(min=0, max=img_h-1)
            x2 = (mlvl_point[:,0] + mlvl_base_len * pos_rho[:, 2]
            ).clamp(min=0, max=img_w-1)
            y2 = (mlvl_point[:,1] + mlvl_base_len * pos_rho[:, 3]
            ).clamp(min=0, max=img_h-1)
            aux_preds = torch.stack([x1, y1, x2, y2], -1)
        det_bboxes = (aux_preds+det_bboxes)/2
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        
        padding = det_scores.new_zeros(det_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        det_scores = torch.cat([det_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(det_bboxes, det_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels