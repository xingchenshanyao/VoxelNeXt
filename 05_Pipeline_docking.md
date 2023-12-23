# Pipeline docking
记录本地管道对接的过程，包括VoxelNeXt与上下游对接

## 一、对接上游
使用nuScenes数据集的原始点云数据与上游给予的伪标签作为训练数据集进行训练

### 1.1. 伪标签格式
参考kitti的txt标签格式，每个bin点云文件对应一个txt标签

由于此时txt标签中，三维框的中心点是基于激光雷达坐标系原点的，所以后续处理时不用进行calib坐标系变化
```
Car [0.80 2.66 1.24] [-0.00 0.16 -0.40] 0
Car [0.43 0.57 9.73] [-54.39 -76.99 -6.57] 0
```
- 第1列（字符串）：代表物体类别（type）——总共有9类，分别是：Car、Pedestrian、Cyclist
- 第2~4列（浮点数）：3D物体的尺寸——分别是高、宽、长（单位：米）
- 第5-7列（浮点数）：3D物体的位置（location）——分别是x、y、z（单位：米），特别注意的是，这里的xyz是在**雷达坐标系**下3D物体的中心点位置
- 第8列（浮点数）：3D物体的空间方向（rotation_y）——取值范围为：-pi ~ pi（单位：rad）

### 1.2 nuScenes数据格式
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
