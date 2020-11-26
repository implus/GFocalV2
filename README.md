# Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection

GFocalV2 (GFLV2) is a next generation of GFocalV1 (GFLV1), which utilizes the statistics of learned bounding box distributions to guide the reliable localization quality estimation.

Again, GFLV2 improves over GFLV1 about ~1 AP without (almost) extra computing cost!

Analysis of GFocalV2 in ZhiHu will come soon. You can first see comments about GFocalV1 in [大白话 Generalized Focal Loss(知乎)](https://zhuanlan.zhihu.com/p/147691786)

[2020.11] GFocalV2 is here!

[2020.11] GFocalV1 has been adopted in [NanoDet](https://github.com/RangiLyu/nanodet), a super efficient object detector on mobile devices, achieving same performance but 2x faster than YoLoV4-Tiny! More details are in [YOLO之外的另一选择，手机端97FPS的Anchor-Free目标检测模型NanoDet现已开源~](https://zhuanlan.zhihu.com/p/306530300).

[2020.10] Good News! GFocalV1 has been accepted in NeurIPs 2020 and GFocalV2 is on the way.

[2020.9] The winner (1st) of GigaVision (object detection and tracking) in ECCV 2020 workshop from DeepBlueAI team adopt GFocalV1 in their [solutions](http://dy.163.com/article/FLF2LGTP0511ABV6.html).

[2020.7] GFocalV1 is officially included in [MMDetection V2](https://github.com/open-mmlab/mmdetection/blob/master/configs/gfl/README.md), many thanks to [@ZwwWayne](https://github.com/ZwwWayne) and [@hellock](https://github.com/hellock) for helping migrating the code.


## Introduction


<img src="https://github.com/implus/GFocalV2/blob/master/gfocal.png" width="1000" height="300" align="middle"/>

For details see [GFocalV2](https://arxiv.org/pdf/2011.12885.pdf). The speed-accuracy trade-off is as follows:

<img src="https://github.com/implus/GFocalV2/blob/master/sota_time_acc.jpg" width="541" height="365" align="middle"/>


## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

## Get Started

Please see [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of MMDetection.

## Train

```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO dataset in 'data/coco/'

./tools/dist_train.sh configs/gfocal/gfocal_r50_fpn_ms2x.py 8 --validate
```

## Inference

```python
./tools/dist_test.sh configs/gfocal/gfocal_r50_fpn_ms2x.py work_dirs/gfocal_r50_fpn_ms2x/epoch_24.pth 8 --eval bbox
```

## Speed Test (FPS)

```python
CUDA_VISIBLE_DEVICES=0 python3 ./tools/benchmark.py configs/gfocal/gfocal_r50_fpn_ms2x.py work_dirs/gfocal_r50_fpn_ms2x/epoch_24.pth
```

## Models

For your convenience, we provide the following trained models (GFocalV2). All models are trained with 16 images in a mini-batch with 8 GPUs.

Model | Multi-scale training | AP (minival) | AP (test-dev) | FPS | Link
--- |:---:|:---:|:---:|:---:|:---:
GFocal_R_50_FPN_1x              | Yes | 41.0 | 41.1 | 19.4 | [Google](https://drive.google.com/file/d/1wSE9-c7tcQwIDPC6Vm_yfOokdPfmYmy7/view?usp=sharing)
GFocal_R_50_FPN_2x              | Yes | 43.9 | 44.4 | 19.4 | [Google](https://drive.google.com/file/d/17-1cKRdR5J3SfZ9NBCwe6QE554uTS30F/view?usp=sharing)
GFocal_R_101_FPN_2x             | Yes | 45.8 | 46.0 | 14.6 | [Google](https://drive.google.com/file/d/1qomgA7mzKW0bwybtG4Avqahv67FUxmNx/view?usp=sharing)
GFocal_R_101_dcnv2_FPN_2x       | Yes | 48.0 | 48.2 | 12.7 | [Google](https://drive.google.com/file/d/1xsBjxmqsJoYZYPMr0k06X5K9nnPrexcx/view?usp=sharing)
GFocal_X_101_dcnv2_FPN_2x       | Yes | 48.8 | 49.0 | 10.7 | [Google](https://drive.google.com/file/d/1AHDVQoclYPSP0Ync2a5FCsr_rhq2QdMH/view?usp=sharing)
GFocal_R2_101_dcnv2_FPN_2x      | Yes | 49.9 | 50.5 | 10.9 | [Google](https://drive.google.com/file/d/1sAXfYLXIxZgMrC44LBqDgfYImThZ_kud/view?usp=sharing)

[0] *The reported numbers here are from new experimental trials (in the cleaned repo), which may be slightly different from the original paper.* \
[1] *Note that the 1x performance may be slightly unstable due to insufficient training. In practice, the 2x results are considerably stable between multiple runs.* \
[2] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[3] *`dcnv2` denotes deformable convolutional networks v2. Note that for ResNe(X)t based models, we apply deformable convolutions from stage c3 to c5 in backbones.* \
[4] *Refer to more details in config files in `config/gfocal/`.* \
[5] *FPS is tested with a single GeForce RTX 2080Ti GPU, using a batch size of 1.* 


## Acknowledgement

Thanks MMDetection team for the wonderful open source project!


## Citation

If you find GFocal useful in your research, please consider citing:

```
@inproceedings{li2020generalized,
    title={Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection},
    author={Li, Xiang and Wang, Wenhai and Wu, Lijun and Chen, Shuo and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
    booktitle={NeurIPS},
    year={2020}
}
```

```
@article{li2020generalizedv2,
    title={Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection},
    author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
    journal={arXiv preprint},
    year={2020}
}
```
