# Debug_VoxelNeXt_demo_test_train
本文为调试代码(细读代码)的记录，按kitti_dataset、nuscenes_dataset、demo、test、train的顺序进行

调试demo、test、train时，配置文件直接调用cbgs_voxel0075_voxelnext.yaml与nuscenes_dataset.yaml

为便于代码分析，调试过程在本地完成
### 部分说明
ubuntu18.04、cuda11.8、python3.8、GPU3070、GPU Driver 520.61.05、torch2.0.0+cu118
***
## 一、demo
原运行指令
```
cd tools
python demo2.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ckpt /home/xingchen/Study/4D_GT/VoxelNeXt/output/nuscenes_models_All/cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_20.pth --data_path /home/xingchen/Study/4D_GT/VoxelNeXt/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025.pcd.bin
```
### 1.1. kitti数据集介绍
