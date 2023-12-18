# VoxelNeXt on the Server
记录VoxelNeXt在服务器上的复现过程
### 部分说明
ubuntu20.04、cuda11.3/11.6、A100*4、GPU Driver510、pytorch1.12.1+cu113
## 一、上传VoxelNeXt程序到服务器内
### a、安装Filezilla
由于服务器无法连接外网(无法直接git clone)，必须通过Filezilla上传程序
```
sudo apt-get install filezilla
```
### b、将VoxelNeXt文件夹传输到服务器指定位置
```
Path = /home/**/**yuan
```
## 二、配置服务器环境
### a、配置torch环境
由于服务器cuda版本为11.6/11.3，无法直接上传本地cuda11.8的环境

此时要么在服务器上装cuda11.8，但是由于显卡驱动是510，最高仅支持cuda11.6，所以装cuda11.8前必须升级驱动

或者重新配置基于cuda11.6的环境，选择后者

在anaconda/envs中拷贝他人环境FN**
```
cd /home/**/anaconda/envs
cp -r FN** VoxelNeXt
```
修改VoxelNeXt/bin/pip，用vim指令打卡编辑
```
cd VoxelNeXt/bin
vim pip
```
打开pip配置文件后，按i进入编辑模式，将第一行改为
```
#!/home/**/anaconda3/envs/VoxelNeXt/bin/python
```
按esc退出编辑模式，按 :x 保存退出pip文件

激活VoxelNeXt并使用pip list查看当前配置
```
conda activate VoxelNeXt
pip list
```
得知
```
torch                     1.12.1+cu113
```
此时的环境是一个支持其他模型的cuda11.3的pytorch1.12.1环境，依赖库肯定不全，需要补充
```
pip install av2==0.2.0
```
```
pip install kornia==0.6.8
```
#### BUG1
使用pip list输出pip列表后，报错
```
WARNING: The repository located at ** is not a trusted or secure host and is being ignored. If this repository is available via HTTPS we recommend you use HTTPS instead, otherwise you may silence this warning and allow it anyway with '--trusted-host **'.
```
根据服务器GPU操作说明中，需要添加下载源信赖命令
```
pip config set global.index-url http:****
pip config set install.trusted-host **
```
### b、安装spconv-cu113
```
pip install spconv-cu113
```
### c、运行setup文件安装部分依赖库
```
cd /home/**/**yuan
python setup.py develop
```
#### BUG2
运行步骤c报错
```
error: Could not find suitable distribution for Requirement.parse('SharedArray')
```
尝试直接安装ShareArray
```
pip install ShareArray
```
失败，尝试通过requirements.txt文件安装
```
pip install -r requirements.txt
```
成功，再次运行python setup.py develop，BUG解决
## 三、使用nuScenes数据集复现
### a、准备nuScenes数据集
服务器中已有nuScenes数据集，查看知nuScenes数据集文件夹结构，改成官网要求的格式
```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```
在VoxelNeXt/data文件夹下创建nuscenes等文件夹，并建立软连接
```
ln -s /data/nuscenes/samples /data/XZY_nuscenes/v1.0-trainval
ln -s /data/nuscenes/sweeps /data/XZY_nuscenes/v1.0-trainval
ln -s /data/nuscenes/maps /data/XZY_nuscenes/v1.0-trainval
ln -s /data/nuscenes/v1.0-trainval /data/XZY_nuscenes/v1.0-trainval
mkdir nuscenes
ln -s /data/XZY_nuscenes/v1.0-trainval /home/**/VoxelNeXt/data/nuscenes
```
为方便测试DEBUG，也将v1.0-mini上传并建立软连接
```
ln -s /data/nuscenes/v1.0-mini /home/**/VoxelNeXt/data/nuscenes
```
如使用nuscense_mini数据集，要修改VoxelNeXt/tools/cfgs/dataset_configs/nuscenes_dataset.yaml 中
```
VERSION: 'v1.0-trainval'
```
为
```
VERSION: 'v1.0-mini'
```
下面称 v1.0-trainval 为 all，v1.0-mini 为 mini
### b、数据集初始化
按要求安装nuscenes-devkit==1.0.5
```
pip install nuscenes-devkit==1.0.5
```
仅跑lidar需要运行
```
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval
# if use v1.0-mini
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-mini
```
#### BUG3
报错
```
AssertionError: Database version not found: /home/**/VoxelNeXt/data/nuscenes/v1.0-trainval/v1.0-trainval
```
原因是步骤a中nuScenes数据集的文件夹结构没有设置好，需要检查重新设置
#### BUG4
```
OSError: 1090 requested and 0 written
```
本地验证后得知，服务器的空间不够了不允许再生成新的文件，调整数据集加载地址到/data

跑multi-modal需要运行
```
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval --with_cam
```
## 四、开始训练
推荐服务器多显卡跑nuScenes数据集，修改tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml中的BATCH_SIZE_PER_GPU=16
```
cd tools
#shell script
python train.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml  # 可以跑但是很慢
bash scripts/dist_train.sh 4 --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml # 可以跑通 # 4是显卡数量
```
预计训练时间为30小时，未避免由于断网等因素终端，选择使用screen命令运行

这样即使关闭本地电脑，服务器依旧能够运行
```
screen -S test # 创建一个新的窗口
cd tools
bash scripts/dist_train.sh 4 --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml # 运行程序
```
ctrl+a，然后按d即可退出screen，服务器仍会在后台运行程序
```
screen -ls # 查看当前窗口任务
```
输出567**.test
```
screen -r 567** # 重新进入窗口
screen -X -S 567**.test quit # 删除窗口
```
更多命令参考：

https://blog.csdn.net/weixin_44612221/article/details/126279971

https://blog.csdn.net/carefree2005/article/details/122415714
![1](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/80ed1933-6055-4300-989c-7a8c6dcbed30)
#### BUG8
重新训练时出现错误
```
RuntimeError: DataLoader worker (pid 819355) is killed by signal: Terminated. ……
```
将output/nuscenes_models文件夹改名，就能重新训练了
## 五、开始测试
```
bash scripts/dist_test.sh NUM_GPUS --cfg_file PATH_TO_CONFIG_FILE --ckpt PATH_TO_MODEL
# 举例
bash scripts/dist_test.sh 4 --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ckpt ~/code/xuzeyuan/VoxelNeXt/output/nuscenes_models/cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_20.pth
```
在mini数据集测试服务器用all训练20轮出的权重
![2023-12-17 15-39-24屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/8f7d1962-4e16-4e84-a425-6fb30dc3021d)
在all数据集测试服务器用all训练20轮出的权重
![test](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/078d69b6-1aae-40a9-aa70-b7b5550a090a)
#### BUG5
![BUG5](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/0c198ce9-ef4d-4f3f-9aee-77d8f4a6cbe5)
```
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
```
numpy库版本过高，解决措施降低numpy库(此时一大堆库的版本也需要更改)或修改代码，选择后者

根据提示打开错误文件
```
vim /home/cbdes/anaconda3/envs/VoxelNeXt/lib/python3.9/site-packages/nuscenes/eval/detection/algo.py
```
将
```
tp = np.cumsum(tp).astype(np.float)
fp = np.cumsum(fp).astype(np.float)
```
改为
```
try:
  tp = np.cumsum(tp).astype(np.float)
  fp = np.cumsum(fp).astype(np.float)
except:
  tp = np.cumsum(tp).astype(np.float64)
  fp = np.cumsum(fp).astype(np.float64)
```
#### BUG6
```
torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 740615) of binary: /home/**/anaconda3/envs/VoxelNeXt/bin/python
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: **
```
多显卡设置的问题，在解决BUG5之后未弹出，暂时忽略
### 六、可视化
可以直接看下一节 重新可视化
```
python demo.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ckpt ~/code/xuzeyuan/VoxelNeXt/output/nuscenes_models/cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_20.pth --data_path /data/XZY_nuscenes/data/nuscenes/v1.0-trainval/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin
# or
python demo.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ckpt ~/code/xuzeyuan/VoxelNeXt/output/nuscenes_models_20231215/cbgs_voxel0075_voxelnext/default/ckpt/voxelnext_nuscenes_kernel1.pth --data_path /data/XZY_nuscenes/data/nuscenes/v1.0-trainval/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin
```
or
```
python demo.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ckpt ~/code/xuzeyuan/VoxelNeXt/output/nuscenes_models_20231215/cbgs_voxel0075_voxelnext/default/ckpt/voxelnext_nuscenes_kernel1.pth --data_path /data/XZY_nuscenes/data/kitti/testing/002323.bin
```
![2222](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/7b002fee-4bc1-44b0-b957-2b9e14a3ffb6)
![2023-12-18 09-48-10屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/35cce7b7-b431-4b2b-b632-289a717742a9)
#### BUG7
```
File "/home/cbdes/code/xuzeyuan/VoxelNeXt/tools/demo.py", line 112, in <module>
    main()
  File "/home/cbdes/code/xuzeyuan/VoxelNeXt/tools/demo.py", line 94, in main
    for idx, data_dict in enumerate(demo_dataset):
  File "/home/cbdes/code/xuzeyuan/VoxelNeXt/tools/demo.py", line 59, in __getitem__
    data_dict = self.prepare_data(data_dict=input_dict)
  File "/home/cbdes/code/xuzeyuan/VoxelNeXt/pcdet/datasets/dataset.py", line 180, in prepare_data
    data_dict = self.point_feature_encoder.forward(data_dict)
  File "/home/cbdes/code/xuzeyuan/VoxelNeXt/pcdet/datasets/processor/point_feature_encoder.py", line 29, in forward
    data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
  File "/home/cbdes/code/xuzeyuan/VoxelNeXt/pcdet/datasets/processor/point_feature_encoder.py", line 48, in absolute_coordinates_encoding
    assert points.shape[-1] == len(self.src_feature_list)
AssertionError
```
可以尝试使用pdb方法DEBUG
```
import pdb; pdb.set_trace()
```
在CSDN与GITHUB上搜索，有许多人遇到同样的问题，dome.py文件与nuScenes数据集的接口做的不对

nuScenes数据集bin点云里没有时间戳这一项，而demo.py里要求有

解决措施，修改demo.py第48行
```
points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
```
为
```
points = np.fromfile(self.sample_file_list[index],dtype=np.float32).reshape(-1, 5)
# or
points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
points = np.c_[points, np.zeros(points.shape[0])]
```
出现新的BUG
```
File "demo.py", line 115, in <module>
    main()
  File "demo.py", line 103, in main
    V.draw_scenes(
  File "/home/xingchen/Study/4D_GT/VoxelNeXt/tools/visual_utils/open3d_vis_utils.py", line 70, in draw_scenes
    vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)
  File "/home/xingchen/Study/4D_GT/VoxelNeXt/tools/visual_utils/open3d_vis_utils.py", line 109, in draw_box
    line_set.paint_uniform_color(box_colormap[ref_labels[i]])
IndexError: list index out of range
```
修改tools/visual_utils/open3d_vis_utils.py中109行
```
line_set.paint_uniform_color(box_colormap[ref_labels[i]])
```
为
```
try:
  line_set.paint_uniform_color(box_colormap[ref_labels[i]])
except:
  continue
```
到此，本地运行就可以得到可视化结果，不过感觉结果不佳，服务器上出现新BUG
```
[Open3D WARNING] GLFW Error: X11: The DISPLAY environment variable is missing
[Open3D WARNING] Failed to initialize GLFW
Traceback (most recent call last):
File "/home/cbdes/code/xuzeyuan/VoxelNeXt/tools/demo.py", line 112, in <module>
    main()
  File "/home/cbdes/code/xuzeyuan/VoxelNeXt/tools/demo.py", line 100, in main
    V.draw_scenes(
  File "/home/cbdes/code/xuzeyuan/VoxelNeXt/tools/visual_utils/open3d_vis_utils.py", line 49, in draw_scenes
    vis.get_render_option().point_size = 1.0
AttributeError: 'NoneType' object has no attribute 'point_size'
```
搜索得知是服务器上没有可视化工具，需要可视化的时候在本地可视化吧

参考来源：

https://github.com/dvlab-research/VoxelNeXt/issues/15

https://blog.csdn.net/jin15203846657/article/details/123087367

https://blog.csdn.net/weixin_44287798/article/details/126925297

## 七、重新可视化
使用demo.py进行可视化的效果很差，最好改写一个可视化demo.py

有人在官方仓库里提到这个问题，官方给了一个demo(详见https://github.com/dvlab-research/VoxelNeXt/issues/15)

但是感觉是后处理的，本地运行报错无法加载open3d.ml.torch
```
Exception: Version mismatch: Open3D needs PyTorch version 1.13.*, but version 2.0.0+cu118 is installed!
```
修改PyTorch版本太麻烦了，暂时跳过这个demo

最后的解决措施

将 tools/cfgs/dataset_configs/nuscenes_dataset.yaml 中60行
```
used_feature_list: ['x', 'y', 'z', 'intensity', 'timestamp'],
src_feature_list: ['x', 'y', 'z', 'intensity', 'timestamp'],
```
改为
```
used_feature_list: ['x', 'y', 'z', 'intensity'],
src_feature_list: ['x', 'y', 'z', 'intensity'],
```
然后在 tools/demo.py 中 48行
```
points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
```
改为
```
points = np.fromfile(self.sample_file_list[index],dtype=np.float32).reshape(-1, 5) # 读取nuScenes .bin文件时需要按包含时间戳的格式读取
points = np.delete(points, 4, axis=1) # 然后删除时间戳这一列数据
```
本地可视化结果(服务器上无法可视化)
```
python demo2.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ckpt /home/xingchen/Study/4D_GT/VoxelNeXt/output/nuscenes_models0/cbgs_voxel0075_voxelnext/default/ckpt/voxelnext_nuscenes_kernel1.pth --data_path /home/xingchen/Study/4D_GT/VoxelNeXt/data/nuscenes/v1.0-mini/sweeps/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603847842.pcd.bin
or
python demo2.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ckpt /home/xingchen/Study/4D_GT/VoxelNeXt/output/nuscenes_models_All/cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_20.pth --data_path /home/xingchen/Study/4D_GT/VoxelNeXt/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025.pcd.bin
```
![2023-12-18 14-50-15屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/5c67b92e-e3c2-40d7-8b9b-589e2225be6b)
![2023-12-18 14-57-26屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/3ab23c88-0a52-48ac-9c4f-7deb1a4f8477)
