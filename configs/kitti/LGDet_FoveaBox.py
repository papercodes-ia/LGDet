_base_ = './fovea_r50_fpn_4x4_1x.py'
model = dict(
    bbox_head=dict(
        type='FoveaHeadLGDet',
        num_classes=1,
        with_deform=True,
        pretrain_mask=False,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    test_cfg=dict(nms=dict(type='nms', iou_threshold=0.7)))
# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=12)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(lr=0.005)
lr_config = dict(warmup='linear', 
                 warmup_ratio=1.0 / 3,
                 step=[8, 11])
checkpoint_config = dict(interval=1)
work_dir = 'work_dirs/kitti/LGDet_FoveaBox'