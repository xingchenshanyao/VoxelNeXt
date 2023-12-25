# Pipeline docking
记录本地管道对接的过程，包括VoxelNeXt与上下游对接

## 一、准备工作
### 1.1. 伪标签格式
参考kitti的txt标签格式，每个bin点云文件对应一个txt标签

由于此时txt标签中，三维框的中心点是基于激光雷达坐标系原点的，所以后续处理时不用进行calib坐标系变化
```
0 [-0.00 0.16 -0.40] [0.80 2.66 1.24] 0
1 [-54.39 -76.99 -6.57] [0.43 0.57 9.73] 1.57
```
- 第1列（整数）：代表物体类别（type）——总共有3类，分别是：**Car:0、Pedestrian:1、Cyclist:2**等
- 第2~4列（浮点数）：3D物体的位置（location）——分别是x、y、z（单位：米），特别注意的是，这里的xyz是在**雷达坐标系**下3D物体的中心点位置
- 第5-7列（浮点数）：3D物体的尺寸——分别是高、宽、长（单位：米）
- 第8列（浮点数）：3D物体的空间方向（rotation），为目标前进方向与激光雷达坐标系+x轴的夹角——取值范围为：-pi ~ pi（单位：rad）

### 1.2. nuScenes数据格式
3D标签保存在sample_annotation.json中，

- sample_annotation：用于标注某个目标在一个sample中方向等信息的三维标注框 # 标注时应该是照着图片在点云文件里标的
```
{
"token": "70aecbe9b64f4722ab3c230391a3beb8", # 每个3D标注框的唯一标识符 每个关键帧大约有50-80个
"sample_token": "cd21dbfc3bd749c7b10a5c42562e0c42", # 此3D框所在的**样本/关键帧**的唯一标识符，普通帧没有标注信息
"instance_token": "6dd2cbf4c24b4caeb625035869bca7b5", # 样本标注所属实例（object instance）的唯一标识符
"visibility_token": "4", # 样本标注的可见性（visibility）的唯一标识符
"attribute_tokens": [ # 样本标注的属性（attribute）（描述运动状态等）的唯一标识符列表
"4d8821270b4a47e3a8a300cbec48188e" 
],
"translation": [ # 样本标注的位置信息，包括 x、y、z 坐标（以米为单位）
373.214,
1130.48,
1.25
],
"size": [ # 样本标注的尺寸信息，包括长、宽、高（以米为单位）
0.621,
0.669,
1.642
],
"rotation": [ # 样本标注的姿态信息，包括四元数表示的旋转
0.9831098797903927,
0.0,
0.0,
-0.18301629506281616
],
"prev": "a1721876c0944cdd92ebc3c75d55d693", # 前者的标记符
"next": "1e8e35d365a441a18dd5503a0ee1c208", # 后者的标记符
"num_lidar_pts": 5, # 激光雷达的点数
"num_radar_pts": 0 # 毫米波雷达的点数
},
```
rotation四元数转化角度值
```python
import math
def EulerAndQuaternionTransform(intput_data):
    data_len = len(intput_data)
    angle_is_not_rad = True # True:角度值 False:弧度制
    if data_len == 3: # 角度值转四元数
            r = 0
            p = 0
            y = 0
            if angle_is_not_rad: # 180 ->pi
                r = math.radians(intput_data[0]) 
                p = math.radians(intput_data[1])
                y = math.radians(intput_data[2])
            else:
                r = intput_data[0] 
                p = intput_data[1]
                y = intput_data[2]
     
            sinp = math.sin(p/2)
            siny = math.sin(y/2)
            sinr = math.sin(r/2)
     
            cosp = math.cos(p/2)
            cosy = math.cos(y/2)
            cosr = math.cos(r/2)
     
            w = cosr*cosp*cosy + sinr*sinp*siny
            x = sinr*cosp*cosy - cosr*sinp*siny
            y = cosr*sinp*cosy + sinr*cosp*siny
            z = cosr*cosp*siny - sinr*sinp*cosy
            return [w,x,y,z]
    elif data_len == 4: # 四元数转角度值
        w = intput_data[0] 
        x = intput_data[1]
        y = intput_data[2]
        z = intput_data[3]
 
        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(2 * (w * y - z * x))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
 
        if angle_is_not_rad : # pi -> 180
            r = math.degrees(r)
            p = math.degrees(p)
            y = math.degrees(y)
        return [r,p,y]

input_data = [0.8391977432162915, 0.0, 0.0, 0.5438263948914979]
# output_data = [0.0, 0.0, 1.1499799908438055] # 弧度值pi
# output_data = [0.0, 0.0, 65.88899999984311] # 角度值180
output_data = EulerAndQuaternionTransform(input_data)
print(output_data)
```
### 1.3. demo生成预测框数据格式
```python
pred_dicts = [{'pred_boxes':[9个参数],'pred_scores':[0.2692,...],'pred_labels':[1,...],'pred_ious':[None,...]}]
eg. [{'pred_boxes':tensor([[ 1.2003e+01,  3.4673e+01, -1.9518e-01,  4.5309e+00,  1.9496e+00,1.6281e+00, -1.2367e-01, -9.0418e-05,  4.2779e-05]], device='cuda:0'),'pred_scores':[0.2692],'pred_labels':[1],'pred_ious':[None, None, None, None, None, None]}]
```
在demo.py第103行使用以下代码确定9个参数含义
```python
            # 为确定pred_boxes的9个参数分别是啥：
            # 激光雷达坐标为原点 x(red) y(green) z(blue) 长(x方向) 宽(y方向) 高(z方向) 弧度制表示的与+x,+y,+z夹角(但是后续+y+z夹角被置0了)
            pred_box0 = [ 0,  0, 0, 10,  5,  2, 0, 0, 1]
            pred_box1 = [ -3.19,  -22.85, -1.89, 4.27,  1.83,  1.66, -1.42, 0, 0]
            pred_box2 = [ -3.19,  -22.85, -1.89, 4.27,  1.83,  1.66, -1.42, 1.28, -8.38]
            # Bebug: pred_boxes中的最后一个框不会被画出来
            pred_boxes = [pred_box0,pred_box1,pred_box2]
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            pred_boxes = torch.tensor(pred_boxes).to(device)
            pred_dicts = [{'pred_boxes':pred_boxes,'pred_scores':torch.tensor([0.4,0.5,0.5]).to(device),'pred_labels':torch.tensor([1,1,1]).to(device),'pred_ious':[None, None, None, None, None, None]}]
```
最终确定：
- 坐标系原点：激光雷达中心
- 第1~3个参数：代表预测框中心点的xyz坐标，单位为米
- 第4~6个参数：代表预测框的长宽高，单位为米，长宽高分别是xyz方向上的边长
- 第7~9个参数：代表预测框的旋转角，单位为rad，旋转角分别是与+x+y+z的夹角

雷达坐标系中，x为red(车前进方向右90°)，y为green(车前进方向)，z为blue(垂直地面向上)

![2023-12-24 11-38-59屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/189a744c-052c-4c78-bfc3-c0d2fba8e371)
### 1.4. 伪标签可视化demo
修改demo.py文件，在75行后添加
```python
parser.add_argument('--txt_path', type=str, default='/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/data/kitti/training/label_2/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.txt',
                        help='specify the point cloud data file or directory')
```
注释104行，并添加
```python
pred_boxes, pred_scores, pred_labels= [],[],[]
            with open(args.txt_path, 'r') as f:
                for line in f:
                    line.strip()
                    a = line.split() # 读取列表
                    pred_boxes.append([float(a[1]),float(a[2]),float(a[3]),float(a[4]),float(a[5]),float(a[6]),float(a[7]),0,0])
                    pred_scores.append(1.0)
                    pred_labels.append(int(a[0]))
            # 弥补Bebug: pred_boxes中的最后一个框不会被画出来
            pred_boxes.append([2,2,2,10,10,10,0,0,0])
            pred_scores.append(0.1)
            pred_labels.append(0)
            # 另一种解决方式
            # 在tools/visual_utils/open3d_vis_utils.py第105行中，修改
            # for i in range(gt_boxes.shape[0]-1):
            # 为
            # for i in range(gt_boxes.shape[0]):

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            pred_boxes = torch.tensor(pred_boxes).to(device)
            pred_scores = torch.tensor(pred_scores).to(device)
            pred_labels = torch.tensor(pred_labels).to(device)
            pred_dicts = [{'pred_boxes':pred_boxes,'pred_scores':pred_scores,'pred_labels':pred_labels,'pred_ious':[None, None, None, None, None, None]}]
```
使用伪标签
```
0 -0.00 0.16 -0.40 2.80 2.66 2.24 0
1 -24.39 -26.99 -0.57 1.43 1.57 3.73 1.57
2 10 10 0 1 1 2 1
```
可视化结果为

![3](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/7168916f-a8f0-477c-bade-632bd2bbaee1)

### 1.5. 检测置信度阈值添加
在tools/visual_utils/open3d_vis_utils.py第104行下添加(1.7包含1.6的改动)
```python
        # 添加预测框置信度阈值进行筛选
        confidence_threshold = 0.2
        if score[i] <= confidence_threshold:
            continue
```
未添加置信度阈值前可视化
![1](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/19d217f1-ac73-417e-9bc1-86c8d83b3a77)

添加置信度阈值(confidence_threshold = 0.2)后可视化
![2](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/48900d2d-3864-4e26-a339-7d80bbda2003

### 1.6. 检测NMS添加
为了更好地体现检测效果，对demo的结果进行非极大值抑制NMS(1.7包含1.6的改动)

在tools/visual_utils/open3d_vis_utils.py第104行下添加
```python
threshold,confidence_threshold=0.5,0.2
    # threshold = 0.5 为交并比阈值
    # confidence_threshold = 0.2 为置信度阈值
    gt_boxes, score = non_maximum_suppression(gt_boxes, score, threshold=threshold,confidence_threshold=confidence_threshold) # 非极大值抑制 # 20231225
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gt_boxes = np.array(gt_boxes)
    score = torch.tensor(score).to(device)
```
并增加函数
```python
def non_maximum_suppression(bboxes, scores, threshold=0.5,confidence_threshold = 0.2):
    # 初始化结果列表
    selected_bboxes = []
    selected_scores = []

    # 按照置信度降序排序
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    while len(sorted_indices) > 0:
        # 选择具有最高置信度的边界框
        best_idx = sorted_indices[0]
        selected_bboxes.append(bboxes[best_idx])
        selected_scores.append(scores[best_idx])

        # 计算与已选中边界框的重叠程度
        overlap_indices = []
        for idx in sorted_indices[1:]:
            if scores[idx] <= confidence_threshold: # 筛选出置信度低的
                overlap_indices.append(idx)
                continue

            overlap = compute_overlap(bboxes[best_idx], bboxes[idx]) # 筛选出交并比高的
            if overlap >= threshold:
                overlap_indices.append(idx)

        # 从排序列表中移除重叠的边界框
        sorted_indices = [idx for idx in sorted_indices if idx not in overlap_indices] # 保留满足要求的
        sorted_indices = sorted_indices[1:]

    return selected_bboxes,selected_scores

# 辅助函数：计算两个边界框的重叠程度
def compute_overlap(bbox1, bbox2):
    # 根据具体应用选择适当的重叠度量方式，例如交并比（IoU）
    # 这里仅简单计算重叠面积占较小边界框面积的比例
    x1, y1, w1, h1 = bbox1[0],bbox1[1],bbox1[3],bbox1[4]
    x2, y2, w2, h2 = bbox2[0],bbox2[1],bbox2[3],bbox2[4]

    area1 = w1 * h1
    area2 = w2 * h2

    intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap = intersection_area / min(area1, area2)
    return overlap
```
添加NMS后可视化效果

<table>
    <tr>
            <th>confidence_threshold = 0.2</th>
            <th>confidence_threshold = 0.3</th>
            <th>confidence_threshold = 0.4</th>
    </tr>
    <tr>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/7c6e3169-c186-4988-a83b-d7d2fc8a0e13 /></td>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/d3935f82-356e-45f9-bce8-dea5544f0851/></td>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/a53d8831-cee0-4d86-8cd7-62a089478d84></td>
    </tr>
    <tr>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/b46fa7fd-95a9-42a4-974a-1bb4ce2a4936 /></td>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/1862cf76-90b2-4022-81c2-4421d2182a2e/></td>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/6f1ea0db-319c-4c50-8d98-b17989c2e447/></td>
    </tr>
    <tr>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/45f5f3e8-1406-493f-89eb-ef10340a0304 /></td>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/7bbb4432-d3fb-483a-b5fb-0948bd790668/></td>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/2679a080-4d72-439e-acb4-1ee316ea4300/></td>
    </tr>
</table>
**

## 二、对接上游
使用nuScenes数据集的原始点云数据与上游给予的伪标签作为训练数据集进行训练
### 2.1. 伪标签划分数据集
为使用伪标签进行模型训练，可以选择在划分数据集的时候将伪标签与真值替换，生成新的
```
gt_database_10sweeps_withvelo
nuscenes_infos_10sweeps_train.pkl
nuscenes_infos_10sweeps_val.pkl
nuscenes_dbinfos_10sweeps_withvelo.pkl
```
在pcdet/datasets/nuscenes/nuscenes_utils.py中367行注释掉
```python
            info['gt_boxes'] = gt_boxes[mask, :]
            info['gt_boxes_velocity'] = velocity[mask, :]
            info['gt_names'] = np.array([map_name_from_general_to_detection[name] for name in names])[mask]
```
并添加
```python
            # 修改成伪标签
            false_gt_boxes, false_gt_names = [],[]
            import os
            folder_path = '/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/data/false_gt'  # 替换为伪标签的文件夹路径
            # txt_name = info['lidar_path'].split("/")[-1][:-4]+'.txt' # 伪标签的名称
            txt_name = 'n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.txt' # 仅测试
            dic = {0:'Car',1:'Pedestrian',2:'Cyclist'}
            txt_path = os.path.join(folder_path, txt_name)
            with open(txt_path, 'r') as f:
                for line in f:
                    line.strip()
                    a = line.split() # 读取列表
                    false_gt_boxes.append([float(a[1]),float(a[2]),float(a[3]),float(a[4]),float(a[5]),float(a[6]),float(a[7]),0,0])
                    false_gt_names.append(dic[int(a[0])])
            info['gt_boxes'] = np.array(false_gt_boxes)
            info['gt_boxes_velocity'] = velocity[mask, :]
            info['gt_names'] = false_gt_names
            print("4D_GT/**VoxelNeXt**/pcdet is running !") # 一个影响不大的Bug
```
