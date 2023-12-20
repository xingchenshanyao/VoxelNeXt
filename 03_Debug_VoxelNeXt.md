# Debug_VoxelNeXt
本文为调试代码(细读代码)的记录，按kitti_dataset、nuscenes_dataset、demo、test、train的顺序进行

为便于代码分析，调试过程在本地完成

参考来源：

KITTI数据集介绍：https://blog.csdn.net/zyw2002/article/details/127395975

### 部分说明
ubuntu18.04、cuda11.8、python3.8、GPU3070、GPU Driver 520.61.05、torch2.0.0+cu118
***
## 一、kitti_dataset
### 1.1. kitti数据集介绍
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
image_2 即2号彩色相机所拍摄的图片（.png）

calib对应每一帧的外参（.txt）

label_2是每帧的标注信息（.txt）

velodyne是Velodyne64所得的点云文件（.bin）
#### 1.1.1. calib
calib文件是相机、雷达、惯导等传感器的矫正数据。以“000001.txt”文件为例，内容如下：
```
P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03
P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03
R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01
```
P0~P3：矫正后的相机投影矩阵R^(3*4)

R0_rect：矫正后的相机旋转矩阵R^(3*3)

Tr_velo_to_cam：从雷达到相机0的旋转平移矩阵R^(3*4)

Tr_imu_to_velo：从惯导或GPS装置到相机的旋转平移矩阵R^(3*4)
#### 1.1.2. image_2
image文件以8位PNG格式存储，图集如下：

![2023-12-20 10-02-45屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/a87fde09-6d9c-4841-8c9e-0809b7ef8b05)
#### 1.1.3. label_2
label文件是KITTI中object的标签和评估数据，以“000001.txt”文件为例，包含样式如下：
```
Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 0.47 1.49 69.44 -1.56
Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57
Cyclist 0.00 3 -1.65 676.60 163.95 688.98 193.93 1.86 0.60 2.02 4.59 1.32 45.84 -1.55
DontCare -1 -1 -10 503.89 169.71 590.61 190.13 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 511.35 174.96 527.81 187.45 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 532.37 176.35 542.68 185.27 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 559.62 175.83 575.40 183.15 -1 -1 -1 -1000 -1000 -1000 -10
```
每行代表一个对象，共16列分别表示
- 第1列（字符串）：代表物体类别（type）——总共有9类，分别是：Car、Van、Truck、Pedestrian、Person_sitting、Cyclist、Tram、Misc、DontCare
- 第2列（浮点数）：代表物体是否被截断（truncated）——数值在0（非截断）到1（截断）之间浮动，数字表示指离开图像边界对象的程度
- 第3列（整数）：代表物体是否被遮挡（occluded）——整数0、1、2、3分别表示被遮挡的程度
- 第4列（弧度数）：物体的观察角度（alpha）——取值范围为：-pi ~ pi（单位：rad）
- 第5~8列（浮点数）：物体的2D边界框大小（bbox）——四个数分别是xmin、ymin、xmax、ymax（单位：pixel），表示2维边界框的左上角和右下角的坐标
- 第9~11列（浮点数）：3D物体的尺寸——分别是高、宽、长（单位：米）
- 第12-14列（浮点数）：3D物体的位置（location）——分别是x、y、z（单位：米），特别注意的是，这里的xyz是在相机坐标系下3D物体的中心点位置
- 第15列（弧度数）：3D物体的空间方向（rotation_y）——取值范围为：-pi ~ pi（单位：rad）
- 第16列（浮点数）：检测的置信度（score）
  
  ![image](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/7be4c5d3-4a3c-4a57-89fe-8b1dbd73e894)
#### 1.1.4. velodyne
velodyne文件是激光雷达的测量数据（绕其垂直轴（逆时针）连续旋转），以“000001.bin”文件为例，内容如下：
```
8D 97 92 41 39 B4 48 3D 58 39 54 3F 00 00 00 00 
83 C0 92 41 87 16 D9 3D 58 39 54 3F 00 00 00 00 
2D 32 4D 42 AE 47 01 3F FE D4 F8 3F 00 00 00 00 
37 89 92 41 D3 4D 62 3E 58 39 54 3F 00 00 00 00 
……
```
点云数据以浮点二进制文件格式存储，每行包含8个数据，每个数据由四位十六进制数表示（浮点数），每个数据通过空格隔开。

一个点云数据由四个浮点数数据构成，分别表示点云的x、y、z、r（强度 or 反射值），点云的存储方式如下表所示：
![image](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/b939e45a-45cd-467d-bf59-3ffbb36bb645)

### 1.2. 数据集初始化
原运行命令
```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
点调试出现bug
```
Backend QtAgg is interactive backend. Turning interactive mode on.
```
那就直接看代码吧

### 1.2.1. main()
```python
if __name__ == '__main__':
    # python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos': # 判断运行参数中有没有create_kitti_infos
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2]))) # 打开配置文件kitti_dataset.yaml
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve() # 文件夹绝对路径 # ROOT_DIR = /home/xingchen/Study/4D_GT/VoxelNeXt
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'], # 只加载三个类别
            data_path=ROOT_DIR / 'data' / 'kitti',  # 数据集路径
            save_path=ROOT_DIR / 'data' / 'kitti' # 处理数据集后的保存路径
        )
```
### 1.2.2. create_kitti_infos()
```python
def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False) # KittiDataset()为kitti数据集加载函数
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split) # train_filename = ROOT_DIR/data/kitti/kitti_infos_train.pkl
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split) # val_filename = ROOT_DIR/data/kitti/kitti_infos_val.pkl
    trainval_filename = save_path / 'kitti_infos_trainval.pkl' # trainval_filename = ROOT_DIR/data/kitti/kitti_infos_trainval.pkl
    test_filename = save_path / 'kitti_infos_test.pkl' # test_filename = ROOT_DIR/data/kitti/kitti_infos_test.pkl

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split) # 加载train的序号 # set_split()为数据集分割函数，根据ImageSets中的txt文件加载其中的序号
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True) #
    with open(train_filename, 'wb') as f: # 创建ROOT_DIR/data/kitti/kitti_infos_train.pkl，wb为以二进制写方式打开，只能写
        pickle.dump(kitti_infos_train, f) 
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split) # 加载val的序号
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f: # 创建ROOT_DIR/data/kitti/kitti_infos_val.pkl，并写入
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f: # 创建ROOT_DIR/data/kitti/kitti_infos_trainval.pkl，并写入
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')# 加载test的序号
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f: # 创建ROOT_DIR/data/kitti/kitti_infos_test.pkl，并写入
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split) # 储存train的真值 # create_groundtruth_database()为创建gt_database文件夹，存储数据集真值信息

    print('---------------Data preparation Done---------------')
```
### 1.2.3. KittiDataset()
```

```










***

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
### d、运行setup文件安装部分依赖库
```
python setup.py develop
```
#### DUG1
运行步骤c报错
