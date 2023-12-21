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
为直接debug demo.py，需要修改tools/demo2.py(注意为nuScenes的可视化在demo2.py中)第70行中
```
parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
```
  为
```
parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml',
                        help='specify the config for demo')
parser.add_argument('--data_path', type=str, default='/home/xingchen/Study/4D_GT/VoxelNeXt/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025.pcd.bin',
                        help='specify the point cloud data file or directory')
parser.add_argument('--ckpt', type=str, default='/home/xingchen/Study/4D_GT/VoxelNeXt/output/nuscenes_models_All/cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_20.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
```
### 1.1. main()

