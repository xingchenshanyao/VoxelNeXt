# Debug_VoxelNeXt_kitti_dataset_nuscenes_dataset
本文为调试代码(细读代码)的记录，按kitti_dataset、nuscenes_dataset、demo、test、train的顺序进行

为便于代码分析，调试过程在本地完成

参考来源：

KITTI数据集介绍：https://blog.csdn.net/zyw2002/article/details/127395975

NuScenes数据集介绍：https://blog.csdn.net/qq_47233366/article/details/123450282；https://blog.csdn.net/weixin_43253464/article/details/120669293

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
- 第16列（浮点数）：检测的置信度（score），只有在test里面才有
  
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
```python
class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args: # 第一次传入的参数
            root_path: # ROOT_DIR/data/kitti
            dataset_cfg: # kitti_dataset.yaml
            class_names: # ['Car', 'Pedestrian', 'Cyclist']
            training: # False
            logger: # None
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode] # DATA_SPLIT: {'train': train,'test': val}
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing') # eg. ROOT_DIR/data/kitti/training

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt') # eg. ROOT_DIR/data/kitti/ImageSets/train.txt
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None 

        self.kitti_infos = []
        self.include_kitti_data(self.mode) # 读取kitti文件夹结构
        # ……
```
### 1.2.4. get_infos()
```python
def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        # 获取具体的文件信息
        def process_single_scene(sample_idx): # sample_idx是文件序号
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label: # 读取标签信息
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts: # 读取点云等信息
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
```
### 1.2.5. create_groundtruth_database()
```python
for k in range(len(infos)): # 读取lidar信息
    print('gt_database sample: %d/%d' % (k + 1, len(infos)))
    info = infos[k]
    sample_idx = info['point_cloud']['lidar_idx']
    points = self.get_lidar(sample_idx)
    annos = info['annos']
    names = annos['name']
    difficulty = annos['difficulty']
    bbox = annos['bbox']
    gt_boxes = annos['gt_boxes_lidar']

    num_obj = gt_boxes.shape[0]
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
    ).numpy()  # (nboxes, npoints)

    for i in range(num_obj):
        filename = '%s_%s_%d.bin' % (sample_idx, names[i], i) # 序号_检测对象名_该检测对象在此文件中的序号
        filepath = database_save_path / filename # eg. ROOT_DIR/data/kitti/gt_database/000000_Pedestrian_0.bin
        gt_points = points[point_indices[i] > 0]

        gt_points[:, :3] -= gt_boxes[i, :3]
        with open(filepath, 'w') as f: # 写入ROOT_DIR/data/kitti/gt_database/000000_Pedestrian_0.bin
            gt_points.tofile(f)

        if (used_classes is None) or names[i] in used_classes:
            db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
            db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                        'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                        'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
            if names[i] in all_db_infos:
                all_db_infos[names[i]].append(db_info)
            else:
                all_db_infos[names[i]] = [db_info]
for k, v in all_db_infos.items():
    print('Database %s: %d' % (k, len(v)))

with open(db_info_save_path, 'wb') as f: # 所有gt写入ROOT_DIR/data/kitti/kitti_dbinfos_train.pkl
    pickle.dump(all_db_infos, f)
```
***
## 二、nuscenes_dataset
### 2.1. nuscenes数据集介绍
完整的nuscenes数据集有300多G，包含850scenes(700trains+150vals)

为便于调试，使用只有4 G和10scenes的nuscenes_mini数据集
```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-mini
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-mini
├── pcdet
├── tools
```
samples中存放传感器（6个相机、1个激光雷达、5个毫米波雷达）所采集到的信息

sweeps中存放的格式与samples是一样的，但是较为次要

maps中存放四张地图照片

 v1.0-mini中都是json文件，存放各种标签信息

- category：表示目标的种类，如汽车
```
{
"token": "1fa93b757fc74fb197cdd60001ad8abf",
"name": "human.pedestrian.adult",
"description": "Adult subcategory."
},
```
- attribute：实例属性，表示同一目标不同状态下的属性描述，如一辆车的移动与停止
```
{
"token": "cb5118da1ab342aa947717dc53544259",
"name": "vehicle.moving",
"description": "Vehicle is moving."
},
```
- visibility：实例的可见性
```
{
"description": "visibility of whole object is between 0 and 40%",
"token": "1",
"level": "v0-40"
},
```
- instance：表示一个实例对象，如某个汽车
```
{
"token": "6dd2cbf4c24b4caeb625035869bca7b5",
"category_token": "1fa93b757fc74fb197cdd60001ad8abf",
"nbr_annotations": 39,
"first_annotation_token": "ef63a697930c4b20a6b9791f423351da",
"last_annotation_token": "8bb63134d48840aaa2993f490855ff0d"
},
```
- sensor：传感器描述，如某个相机
```
{
"token": "725903f5b62f56118f4094b46a4470d8",
"channel": "CAM_FRONT",
"modality": "camera"
},
```
- calibrated_sensor：表示传感器的内外参数等信息
```
{
"token": "f4d2a6c281f34a7eb8bb033d82321f79",
"sensor_token": "47fcd48f71d75e0da5c8c1704a9bfe0a",
"translation": [
3.412,
0.0,
0.5
],
"rotation": [
0.9999984769132877,
0.0,
0.0,
0.0017453283658983088
],
"camera_intrinsic": []
},
```
- ego_pose：表示某个时间车辆的姿态
```
{
"token": "5ace90b379af485b9dcb1584b01e7212",
"timestamp": 1532402927814384,
"rotation": [
0.5731787718287827,
-0.0015811634307974854,
0.013859363182046986,
-0.8193116095230444
],
"translation": [
410.77878632230204,
1179.4673290964536,
0.0
]
},
```
- log：表示提取出数据的日志文件
```
{
"token": "7e25a2c8ea1f41c5b0da1e69ecfa71a2", # 每个日志文件的唯一标识符
"logfile": "n015-2018-07-24-11-22-45+0800", # 日志文件的文件名
"vehicle": "n015", # 与日志文件相关联的车辆的唯一标识符
"date_captured": "2018-07-24", # 采集时间
"location": "singapore-onenorth" # 日志文件记录的车辆位置的全球定位系统（GPS）坐标
},
```
- scenes：来自日志文件中一个20s的连续帧
```
{
"token": "cc8c0bf57f984915a77078b10eb33198", # 每个场景的唯一标识符 共10个
"log_token": "7e25a2c8ea1f41c5b0da1e69ecfa71a2",  # 与场景相关联的日志文件（log.json）的唯一标识符
"nbr_samples": 39, # 场景中的样本数（sample.json 中的样本）
"first_sample_token": "ca9a282c9e77460f8360f564131a8af5", # 场景中第一个样本的唯一标识符
"last_sample_token": "ed5fc18c31904f96a8f0dbb99ff069c0", # 场景中最后一个样本的唯一标识符
"name": "scene-0061", # 场景名
"description": "Parked truck, construction, intersection, turn left, following a van" # 场景描述
},
```
- sample：表示每隔0.5s采集一次的经过标注的关键帧
```
{
"token": "ca9a282c9e77460f8360f564131a8af5",
"timestamp": 1532402927647951,
"prev": "",
"next": "39586f9d59004284a7114a68825e8eec",
"scene_token": "cc8c0bf57f984915a77078b10eb33198"
},
```
- sample_data：传感器返回的数据，如雷达点云或图片
```
{
"token": "5ace90b379af485b9dcb1584b01e7212",
"sample_token": "39586f9d59004284a7114a68825e8eec",
"ego_pose_token": "5ace90b379af485b9dcb1584b01e7212",
"calibrated_sensor_token": "f4d2a6c281f34a7eb8bb033d82321f79",
"timestamp": 1532402927814384,
"fileformat": "pcd",
"is_key_frame": false,
"height": 0,
"width": 0,
"filename": "sweeps/RADAR_FRONT/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402927814384.pcd",
"prev": "f0b8593e08594a3eb1152c138b312813",
"next": "978db2bcdf584b799c13594a348576d2"
},
```
- sample_annotation：用于标注某个目标在一个sample中方向等信息的三维标注框
```
{
"token": "70aecbe9b64f4722ab3c230391a3beb8",
"sample_token": "cd21dbfc3bd749c7b10a5c42562e0c42",
"instance_token": "6dd2cbf4c24b4caeb625035869bca7b5",
"visibility_token": "4",
"attribute_tokens": [
"4d8821270b4a47e3a8a300cbec48188e"
],
"translation": [
373.214,
1130.48,
1.25
],
"size": [
0.621,
0.669,
1.642
],
"rotation": [
0.9831098797903927,
0.0,
0.0,
-0.18301629506281616
],
"prev": "a1721876c0944cdd92ebc3c75d55d693",
"next": "1e8e35d365a441a18dd5503a0ee1c208",
"num_lidar_pts": 5,
"num_radar_pts": 0
},
```
![1](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/8c1165dd-1cb6-45eb-8b87-03405f5ef288)
### 2.2. 数据集初始化
原命令
```
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-mini
```
将pcdet/datasets/nuscenes/nuscenes_dataset第8行、第308中
```
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate

parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
```
改为
```
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate

parser.add_argument('--cfg_file', type=str, default='tools/cfgs/dataset_configs/nuscenes_dataset.yaml', help='specify the config of dataset')
parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
parser.add_argument('--version', type=str, default='v1.0-mini', help='')
```
即可debug调试
#### 2.2.1. main()
```python
if __name__ == '__main__':
    # python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-mini
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/dataset_configs/nuscenes_dataset.yaml', help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='')
    args = parser.parse_args()

    if args.func == 'create_nuscenes_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file))) # 打开配置文件nuscenes_dataset.yaml
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve() # ROOT_DIR = /home/xingchen/Study/4D_GT/VoxelNeXt
        dataset_cfg.VERSION = args.version # dataset_cfg.VERSION = 'v1.0-mini'
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'nuscenes',
            save_path=ROOT_DIR / 'data' / 'nuscenes',
            max_sweeps=dataset_cfg.MAX_SWEEPS, # max_sweeps = 10 # 暂时不知道这是干嘛的
        )

        nuscenes_dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'nuscenes',
            logger=common_utils.create_logger(), training=True
        )
        nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS) # 划分真值

```
#### 2.2.2. create_nuscenes_info()
需要继续改正第257行中
```
from . import nuscenes_utils
```
为
```
from pcdet.datasets.nuscenes import nuscenes_utils
```
```python
def create_nuscenes_info(version, data_path, save_path, max_sweeps=10):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from pcdet.datasets.nuscenes import nuscenes_utils # 一个转化表格，eg:'human.pedestrian.adult' -> 'pedestrian'
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train # train_scenes = ['scene-0061', 'scene-0553',..., 'scene-1100'] # 共8个
        val_scenes = splits.mini_val # val_scenes = ['scene-0103', 'scene-0916']
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True) # NuScenes()用于nuScenes的数据库类，帮助从数据库中查询和检索信息
    available_scenes = nuscenes_utils.get_available_scenes(nusc) # 返回可用的场景 # 10个都可以用
    available_scene_names = [s['name'] for s in available_scenes] # 返回可用的场景的名字 # available_scene_names = ['scene-0061', 'scene-0103', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-0916', 'scene-1077', 'scene-1094', 'scene-1100']
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes)) # ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes)) # ['scene-0103', 'scene-0916']
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes]) # val_scenes = {'325cef682f064c55a255...625c533b75', 'fcbccedd61424f1b85dc...8f897f9754'} # 改成

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps
    ) # 编写nuscenes_infos_10sweeps_train.pkl和nuscenes_infos_10sweeps_val.pkl

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f: # 新建并写入nuscenes_infos_10sweeps_train.pkl
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f: # 新建并写入nuscenes_infos_10sweeps_val.pkl
            pickle.dump(val_nusc_infos, f)

```
####  2.2.3. fill_trainval_infos()
```python
def fill_trainval_infos(data_path, nusc, train_scenes, val_scenes, test=False, max_sweeps=10):
    train_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(nusc.sample), desc='create_info', dynamic_ncols=True)

    ref_chan = 'LIDAR_TOP'  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = 'LIDAR_TOP'  # The reference channel of the current sample_rec that the point clouds are mapped to.

    for index, sample in enumerate(nusc.sample):
        progress_bar.update()

        ref_sd_token = sample['data'][ref_chan] # '9d9bf11fb0e144c8b446d54a8a00184f'
        ref_sd_rec = nusc.get('sample_data', ref_sd_token) # {'token': '9d9bf11fb0e144c8b446...4a8a00184f', ..., 'channel': 'LIDAR_TOP'}
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token']) # {'token': 'a183049901c24361a6b0...1b8013137c', ..., 'camera_intrinsic': []}
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token']) # {'token': '9d9bf11fb0e144c8b446...4a8a00184f', ..., 'translation': [411.3039349319818, 1180.8903791765097, 0.0]}
        ref_time = 1e-6 * ref_sd_rec['timestamp'] # 1532402927.647951

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token) # lidar文件路径，数据信息

        ref_cam_front_token = sample['data']['CAM_FRONT'] # 'e3d495d4ac534d54b321f50006683844'
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token) # camera文件路径，图片信息

        # Homogeneous transform from ego car frame to reference frame # 从自车坐标系到参考坐标系的同质变换
        ref_from_car = transform_matrix(
            ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True
        )

        # Homogeneous transformation matrix from global to _current_ ego car frame # 从全局坐标系到自车坐标系的同质变换
        car_from_global = transform_matrix(
            ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True,
        )

        info = {
            'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(), # lidar的.bin文件路径
            'cam_front_path': Path(ref_cam_path).relative_to(data_path).__str__(), # camera的.jpg文件路径
            'cam_intrinsic': ref_cam_intrinsic, # camrea的**转换矩阵
            'token': sample['token'], # 单个对象(点云+图像)唯一索引
            'sweeps': [], # 
            'ref_from_car': ref_from_car, # 坐标系**转换矩阵
            'car_from_global': car_from_global, # 坐标系**转换矩阵
            'timestamp': ref_time, # 时间戳
        }

        sample_data_token = sample['data'][chan] # '9d9bf11fb0e144c8b446d54a8a00184f'
        curr_sd_rec = nusc.get('sample_data', sample_data_token)
        sweeps = []
        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec['prev'] == '':
                if len(sweeps) == 0:
                    sweep = {
                        'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
                        'sample_data_token': curr_sd_rec['token'],
                        'transform_matrix': None,
                        'time_lag': curr_sd_rec['timestamp'] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])

                # Get past pose
                current_pose_rec = nusc.get('ego_pose', curr_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(
                    current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    'calibrated_sensor', curr_sd_rec['calibrated_sensor_token']
                )
                car_from_current = transform_matrix(
                    current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False,
                )

                tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

                lidar_path = nusc.get_sample_data_path(curr_sd_rec['token'])

                time_lag = ref_time - 1e-6 * curr_sd_rec['timestamp']

                sweep = {
                    'lidar_path': Path(lidar_path).relative_to(data_path).__str__(),
                    'sample_data_token': curr_sd_rec['token'],
                    'transform_matrix': tm,
                    'global_from_car': global_from_car,
                    'car_from_current': car_from_current,
                    'time_lag': time_lag,
                }
                sweeps.append(sweep)

        info['sweeps'] = sweeps

        assert len(info['sweeps']) == max_sweeps - 1, \
            f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, " \
            f"you should duplicate to sweep num {max_sweeps - 1}"

        if not test:
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]

            # the filtering gives 0.5~1 map improvement
            num_lidar_pts = np.array([anno['num_lidar_pts'] for anno in annotations])
            num_radar_pts = np.array([anno['num_radar_pts'] for anno in annotations])
            mask = (num_lidar_pts + num_radar_pts > 0)

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)

            assert len(annotations) == len(gt_boxes) == len(velocity)

            info['gt_boxes'] = gt_boxes[mask, :]
            info['gt_boxes_velocity'] = velocity[mask, :]
            info['gt_names'] = np.array([map_name_from_general_to_detection[name] for name in names])[mask]
            info['gt_boxes_token'] = tokens[mask]
            info['num_lidar_pts'] = num_lidar_pts[mask]
            info['num_radar_pts'] = num_radar_pts[mask]

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos
```
####  2.2.4. create_groundtruth_database()
```python
def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo' # PosixPath('/home/xingchen/Study/4D_GT/VoxelNeXt/data/nuscenes/v1.0-mini/gt_database_10sweeps_withvelo')
        db_info_save_path = self.root_path / f'nuscenes_dbinfos_{max_sweeps}sweeps_withvelo.pkl' # PosixPath('/home/xingchen/Study/4D_GT/VoxelNeXt/data/nuscenes/v1.0-mini/nuscenes_dbinfos_10sweeps_withvelo.pkl')

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx # 0
            info = self.infos[idx] # {'lidar_path': 'samples/LIDAR_TOP/n0...51.pcd.bin', 'cam_front_path': 'samples/CAM_FRONT/n0...612460.jpg', 'cam_intrinsic': array([[1.26641720e+...000e+00]]), 'token': 'ca9a282c9e77460f8360...64131a8af5', 'sweeps': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}], 'ref_from_car': array([[ 0.00203327,...       ]]), 'car_from_global': array([[-3.45552926e...000e+00]]), 'timestamp': 1532402927.647951, 'gt_boxes': array([[ 1.84143850e...306e-03]]), 'gt_boxes_velocity': array([[ 3.07550587e...595e-02]]), 'gt_names': array(['pedestrian',...pe='<U20'), 'gt_boxes_token': array(['ef63a697930c...pe='<U32'), 'num_lidar_pts': array([  1,   2,   5...      27]), 'num_radar_pts': array([ 0,  0,  0,  ...,  1,  2])}
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps) # 去除了某些点，静态点？小于10个点的对象框？
            gt_boxes = info['gt_boxes'] # 66个三维对象框
            gt_names = info['gt_names'] # 66个对象框类别

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy() # box_idxs_of_pts = array([-1, -1, -1, ..., -1, -1, -1]) # -1表示不是背景

            for i in range(gt_boxes.shape[0]): # 分别建立写入66个对象
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i) # '0_pedestrian_0.bin'
                filepath = database_save_path / filename # PosixPath('/home/xingchen/Study/4D_GT/VoxelNeXt/data/nuscenes/v1.0-mini/gt_database_10sweeps_withvelo/0_pedestrian_0.bin')
                gt_points = points[box_idxs_of_pts == i] # array([[1.8647348e+01, 5.9553642e+01, 5.0377835e-02, 3.5000000e+01,        0.0000000e+00],       [1.8647348e+01, 5.9553642e+01, 5.0377835e-02, 3.5000000e+01,        0.0000000e+00],       [1.8647348e+01, 5.9553642e+01, 5.0377835e-02, 3.5000000e+01,        0.0000000e+00],       [1.8647348e+01, 5.9553642e+01, 5.0377835e-02, 3.5000000e+01,        0.0000000e+00],       [1.8647348e+01, 5.9553642e+01, 5.0377835e-02, 3.5000000e+01,        0.0000000e+00],       [1.8647348e+01, 5.9553642e+01, 5.0377835e-02, 3.5000000e+01,        0.0000000e+00],       [1.8647348e+01, 5.9553642e+01, 5.0377835e-02, 3.5000000e+01,        0.0000000e+00],       [1.8647348e+01, 5.9553642e+01, 5.0377835e-02, 3.5000000e+01,        0.0000000e+00],       [1.8647348e+01, 5.9553642e+01, 5.0377835e-02, 3.5000000e+01,        0.0000000e+00],       [1.8647348e+01, 5.9553642e+01, 5.0377835e-02, 3.5000000e+01,        0.0000000e+00]], dtype=float32)

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f: # 
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # 'gt_database_10sweeps_withvelo/0_pedestrian_0.bin'
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]} # {'name': 'pedestrian', 'path': 'gt_database_10sweeps...rian_0.bin', 'image_idx': 0, 'gt_idx': 0, 'box3d_lidar': array([ 1.84143850e+...4613e-04]), 'num_points_in_gt': 10}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f: # 打开<_io.BufferedWriter name='/home/xingchen/Study/4D_GT/VoxelNeXt/data/nuscenes/v1.0-mini/nuscenes_dbinfos_10sweeps_withvelo.pkl'>并写入
            pickle.dump(all_db_infos, f)
```
