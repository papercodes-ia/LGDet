# Learning More from Labels: Regularizing Deep Networks with Label Geometry for Accurate Object Localization

## Installation

Our code is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please install mmdetection first, and then put our code and [KITTI images](http://www.cvlibs.net/datasets/kitti/) (CC BY-NC-SA 3.0) in corresponding folders.

## Training

To train the model in the paper, run this command:

```train
python tools/train.py configs/kitti/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x-LGDet.py
```

We use ResNet-50 pretrained on Imagenet to train our models, which can be downloaded by default when running the training command. For a better performance, we can also pretrain distance transforms with pretrain_mask set to True.

## Evaluation

To evaluate the model on KITTI val, run:

```eval
python tools/test.py configs/kitti/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x-LGDet.py work_dirs/kitti/<path_to_trained_model> --eval bbox
```

## Results

Our model achieves the following performance on KITTI val:

| Model name      | AP70 | AP75 | AP80 | AP85 | AP90 | AP95 | APs70 | APm70 | APl70 | Easy | Moderate | Hard |
|:---------------:|:----:|:----:|:----:|:----:|:----:|:----:|:-----:|:-----:|:-----:|:----:|:--------:|:----:|
| FCOS (baseline) | 92.0 | 88.4 | 81.1 | 66.0 | 37.7 | 4.3  | 84.0  | 92.3  | 95.5  | 91.42| 86.59    | 85.89|
| LGDet (ours)    | 94.6 | 92.1 | 86.8 | 74.4 | 45.6 | 7.6  | 90.2  | 94.6  | 96.4  | 91.29| 87.61    | 88.03|


## License

This project is released under the MIT license.
