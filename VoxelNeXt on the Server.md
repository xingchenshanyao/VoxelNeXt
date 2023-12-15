# VoxelNeXt on the Server
记录VoxelNeXt在服务器上的复现过程
### 部分说明
ubuntu20.04、cuda11.6/11.3、A100*4、GPU Driver510
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
### a、复制现有torch环境
由于服务器cuda版本为11.6/11.3，无法直接上传本地cuda11.8的环境

此时要么在服务器上装cuda11.8，但是由于显卡驱动是510，最高仅支持cuda11.6，所以装cuda11.8前必须升级驱动

或者重新配置cuda11.6的环境，选择后者

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
搜索后得知，之后使用pip安装依赖需要添加'--trusted-host **'，这个BUG暂时没有影响
```
pip install name --trusted-host **
```
### b、安装spconv-cu118
```
pip install spconv-cu113 --trusted-host **
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
解决措施，先根据服务器GPU操作说明中，添加pip下载源信赖命令
```
pip config set global.index-url http:****
pip config set install.trusted-host **
```
然后尝试直接安装ShareArray
```
pip install ShareArray
```
失败，尝试通过requirements.txt文件安装
```
pip install -r requirements.txt
```
成功，再次运行python setup.py develop，未报错，BUG解决
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
mkdir nuscenes
cd nuscenes
mkdir v1.0-trainval
cd v1.0-trainval
ln -s /home/**/data/nuscenes/samples /home/**/VoxelNeXt/data/nuscenes/v1.0-trainval
ln -s /home/**/data/nuscenes/sweeps /home/**/VoxelNeXt/data/nuscenes/v1.0-trainval
ln -s /home/**/data/nuscenes/maps /home/**/VoxelNeXt/data/nuscenes/v1.0-trainval
ln -s /home/**/data/nuscenes/v1.0-trainval /home/**/VoxelNeXt/data/nuscenes/v1.0-trainval
```
退回VoxelNeXt文件夹
```
cd /home/**/VoxelNeXt
```
### b、数据集初始化
按要求安装nuscenes-devkit==1.0.5
```
pip install nuscenes-devkit==1.0.5
```
仅跑lidar需要运行
```
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval
```
#### BUG3
报错
```
AssertionError: Database version not found: /home/**/VoxelNeXt/data/nuscenes/v1.0-trainval/v1.0-trainval
```
原因是步骤a中nuScenes数据集的文件夹结构没有设置好，需要检查重新设置
跑multi-modal需要运行
```
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval --with_cam
```
