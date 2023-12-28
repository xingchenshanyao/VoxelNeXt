# VoxelNeXt
记录复现VoxelNeXt的流程及各种BUG与解决措施

参考来源

https://github.com/dvlab-research/VoxelNeXt

https://blog.csdn.net/AaaA00000001/article/details/127021967

https://blog.csdn.net/weixin_52288941/article/details/133518555

https://blog.csdn.net/weixin_45811857/article/details/124457280

### 部分说明
ubuntu18.04、cuda11.8、python3.8、GPU3070、GPU Driver 520.61.05、torch2.0.0+cu118
## 一、克隆VoxelNeXt仓库并进入文件夹
```
git clone https://github.com/dvlab-research/VoxelNeXt && cd VoxelNeXt
```
## 二、安装OpenPCDet环境
### 0、最便捷的方法
cuda11.8版本一样的话，直接到我这里拷贝环境
### a、复制现有torch环境作为基础
例如，将anaconda3/envs/torch20230517(一个能跑通YOLOP的torch环境)文件夹另存副本，改名为VoxelNeXt

然后更改anaconda3/envs/VoxelNeXt/bin/pip中的
```
#!/home/xingchen/anaconda3/envs/torch20230517/bin/python
```
为
```
#!/home/xingchen/anaconda3/envs/VoxelNeXt/bin/python
```
### b、安装pcdet v0.5
```
git clone https://github.com/open-mmlab/OpenPCDet.git
```
### c、安装spconv-cu118
```
pip install spconv-cu118
```
其他cuda版本参照
```
https://github.com/traveller59/spconv
```
### d、运行setup文件安装部分依赖库
```
python setup.py develop
```
#### DUG1
运行步骤c报错
```
The detected CUDA version (9.1) mismatches the version that was used to compile PyTorch (11.8). Please make sure to use the same CUDA versions.
```
说该程序只能找到cuda9.1，找不到cuda11.8

终端运行nvidia-smi显示cuda11.8，但是运行nvcc -V显示cuda9.1

搜索后得知，官网下载和离线下载cuda，cuda安装方式不一样，nvcc -V搜索不到离线安装的cuda11.8，步骤c同理

解决方法，在官网重新安装cuda11.8，并在.bashrc中添加cuda11.8路径
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```
![2023-12-13 16-55-16屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/e668c6e9-03a3-4f7e-bff5-6981541ba846)
其他cuda版本参照
```
https://developer.nvidia.com/cuda-toolkit-archive
```
安装cuda11.8时的具体操作步骤参照下文中：4.到cuda-toolki-archive,下载对应的cuda

https://blog.csdn.net/weixin_45811857/article/details/124457280

主目录ctrl+H可见隐藏的.bashrc文件，打开，注释掉
```
# export PATH=/usr/local/cuda/bin:$PATH
# export C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
# export PKG_CONFIG_PATH=/usr/local/cuda/pkgconfig:${PKG_CONFIG_PATH}
```
添加
```
export PATH=/usr/local/cuda-11.8/bin:$PATH
export C_INCLUDE_PATH=/usr/local/cuda-11.8/include:${C_INCLUDE_PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=/usr/local/cuda-11.8/pkgconfig:${PKG_CONFIG_PATH}
```
解决BUG1，可顺利运行setup文件
### e、运行requirements.txt文件安装部分依赖库
```
pip install -r requirements.txt
```
#### DUG2
跑起来的时候发现还缺部分依赖库
```
No module named 'kornia'
No module named 'av2'
No module named 'mayavi'
```
解决措施
```
pip install kornia==0.6.8 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```
```
pip install av2==0.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```
```
pip install mayavi -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```
demo.py中需要调用mayavi或者open3D，如果mayavi安装失败就用open3D代替
```
pip install open3D -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```
## 三、准备KITTI数据集
VoxelNeXt支持多种数据集，本次复现使用KITTI
### a、下载KITTI
官网或个人网盘下载KITTI数据集、以及KITTI数据集划分文件，参考下文中：1 准备工作 and 1.1 创建数据目录

https://blog.csdn.net/qq_46127597/article/details/123291204

数据集文件夹结构
```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │   ├──test.txt & train.txt & trainval.txt & val.txt
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```
VoxelNeXt官网中对KITTI数据集中做了road plane的扩充，可以选择下载

https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md
### b、数据集初始化
```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
#### DUG3
运行步骤b报错
```
KeyError: ‘road_plane’
```
如果步骤a中没有选择下载road plane，那么需要更改tools/cfgs/kitti_models/voxelnext.yaml中
```
USE_ROAD_PLANE: True
```
为
```
USE_ROAD_PLANE: False
```
或者在tools/cfgs/dataset_configs/kitti_dataset.yaml中
## 四、训练KITTI数据集
### a、设置训练参数
在tools/cfgs/kitti_models/voxelnext.yaml中，根据自己的显卡更改参数，以3070+8G显存为例
```
BATCH_SIZE_PER_GPU: 4 # 20231213 # 每块GPU的batchseize
NUM_EPOCHS: 80 # 训练轮次
```
### b、开始训练
3070+batchsize4+80epoches训练时长约为7小时
```
cd tools
#shell script
python train.py --cfg_file cfgs/kitti_models/voxelnext.yaml # 这个单显卡跑通了
sh scripts/dist_train.sh 4 --cfg_file cfgs/kitti_models/voxelnext.yaml # 多显卡还没试 # 4是显卡数量
```
## 五、测试KITTI数据集
```
cd tools
#shell script
python test.py --cfg_file cfgs/kitti_models/voxelnext.yaml --batch_size 4 --ckpt output/kitti_models/voxelnext/default/ckpt/checkpoint_epoch_80.pth # 单显卡测试
```
## 六、可视化
```
python demo.py --cfg_file cfgs/kitti_models/voxelnext.yaml --ckpt /home/xingchen/Study/4D_GT/VoxelNeXt/output/kitti_models/voxelnext/default/ckpt/checkpoint_epoch_80.pth --data_path /home/xingchen/Study/4D_GT/VoxelNeXt/data/kitti/testing/velodyne/000010.bin
```
![2023-12-13 19-57-26屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/1657c637-b3d3-4a8b-b80e-df5cd7e21241)
