# Rasterization
将VoxelNext的预测结果或真值在点云数据中按散点上色并可视化

## 一、预测结果上色
所有预测框内的散点显示为红色，修改tools/visual_utils/open3d_vis_utils.py
```python
"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import open3d as o3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    threshold,confidence_threshold=0.5,0.2
    # threshold = 0.5 为交并比阈值
    # confidence_threshold = 0.2 为置信度阈值
    ref_boxes, ref_scores = non_maximum_suppression(ref_boxes, ref_scores, threshold=threshold,confidence_threshold=confidence_threshold) # 非极大值抑制 # 20231225
    
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        
        points = points[:, :3]
        color = np.ones_like(points) * [1, 1, 1] # 初始颜色为白色
        # confidence_threshold = 0.2 # 上色的置信度
        for i in range(len(ref_boxes)):
            box = ref_boxes[i]
            label = ref_labels[i]
            # score = ref_scores[i]
            # if score <=confidence_threshold:
            #     continue
            color = add_color(box,label,points,color)
        pts.colors = open3d.utility.Vector3dVector(color)

        # # 为点云上色
        # color = [1.0, 0.0, 0.0]  # 设置颜色为红色
        # pts.paint_uniform_color(color)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    # threshold,confidence_threshold=0.5,0.2
    # # threshold = 0.5 为交并比阈值
    # # confidence_threshold = 0.2 为置信度阈值
    # gt_boxes, score = non_maximum_suppression(gt_boxes, score, threshold=threshold,confidence_threshold=confidence_threshold) # 非极大值抑制 # 20231225
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gt_boxes = np.array(gt_boxes)
    score = torch.tensor(score).to(device)


    for i in range(gt_boxes.shape[0]):
        # # 添加预测框置信度阈值进行筛选 # 20231225
        # confidence_threshold = 0.2
        # if score[i] <= confidence_threshold:
        #     continue
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            # line_set.paint_uniform_color(box_colormap[ref_labels[i]])
            try:
                line_set.paint_uniform_color(box_colormap[ref_labels[i]])
            except:
                continue

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

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


def rotate_points(points, center, rotation_angle):
    """
    将点云中的点围绕指定中心点进行水平旋转
    
    参数：
    points: 点云数据，类型为 open3d.utility.Vector3dVector
    center: 旋转中心点的坐标，形状为 (3,)
    rotation_angle: 旋转角度（弧度）
    
    返回值：
    rotated_points: 旋转后的点云数据，类型为 open3d.utility.Vector3dVector
    """
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                [0, 0, 1]])
    translated_points = np.asarray(points) - center
    rotated_points = np.dot(rotation_matrix, translated_points.T).T
    rotated_points += center
    return o3d.utility.Vector3dVector(rotated_points)

def color_points_in_bbox(points, bbox_center, bbox_size, bbox_rotation, color_in_bbox,color):
    """
    将点云中某个有水平旋转角度的边界框范围内的点上色
    
    参数：
    points: 点云数据，类型为 open3d.utility.Vector3dVector
    bbox_center: 边界框的中心坐标，形状为 (3,)
    bbox_size: 边界框的尺寸，形状为 (3,)
    bbox_rotation: 边界框的旋转角度（弧度）
    color_in_bbox: 边界框内点的颜色
    color_outside_bbox: 边界框外点的颜色
    
    返回值：
    colored_points: 上色后的点云数据，类型为 open3d.utility.Vector3dVector
    """
    rotated_points = rotate_points(points, bbox_center, bbox_rotation)
    bbox_min = bbox_center - bbox_size / 2
    bbox_max = bbox_center + bbox_size / 2
    inside_bbox = (np.all(rotated_points >= bbox_min, axis=1) & np.all(rotated_points <= bbox_max, axis=1))
    color[inside_bbox] = color_in_bbox
    return color

def add_color(box,label,points,color):
    bbox_center = np.array([box[0], box[1], box[2]])
    bbox_size = np.array([box[3], box[4], box[5]])
    bbox_rotation = box[6] 
    if label >= 0:
        color_in_bbox = np.array([1, 0, 0])  # 红色
    color = color_points_in_bbox(points, bbox_center, bbox_size, bbox_rotation, color_in_bbox,color)
    return color
```
可视化结果
![1](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/373c18c7-b023-4f67-9672-0da95ee6544d)
![2](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/62f4390f-0850-41f6-8fc5-d4fef44e65d9)

## 二、按类别上不同颜色
修改tools/visual_utils/open3d_vis_utils.py中
```python
def add_color(box,label,points,color):
    bbox_center = np.array([box[0], box[1], box[2]])
    bbox_size = np.array([box[3], box[4], box[5]])
    bbox_rotation = box[6] 
    if label >= 0:
        color_in_bbox = np.array([1, 0, 0])  # 红色
    color = color_points_in_bbox(points, bbox_center, bbox_size, bbox_rotation, color_in_bbox,color)
    return color
```
为
```python
# box_colormap = [
#     [1, 1, 1],
#     [0, 1, 0],
#     [0, 1, 1],
#     [1, 1, 0],]
def add_color(box,label,points,color):
    bbox_center = np.array([box[0], box[1], box[2]])
    bbox_size = np.array([box[3], box[4], box[5]])
    bbox_rotation = box[6] 
    try:
        color_in_bbox = box_colormap[label]
    except:
        color_in_bbox = np.array([1, 1, 1])
    # if label == 1:
    #     color_in_bbox = np.array([1, 0, 0])  # 红色
    # else:
    #     color_in_bbox = np.array([0, 1, 0])  # 绿色
    color = color_points_in_bbox(points, bbox_center, bbox_size, bbox_rotation, color_in_bbox,color)
    return color
```
可视化结果1
```
python demo2.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ckpt /home/xingchen/Study/4D_GT/VoxelNeXt/output/nuscenes_models_All/cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_20.pth --data_path /home/xingchen/Study/4D_GT/VoxelNeXt/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025.pcd.bin
```
![1](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/1367c215-b6a4-49d7-978e-4275ec92d3f0)
可视化结果2
```
python demo2.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ckpt /home/xingchen/Study/4D_GT/VoxelNeXt/output/nuscenes_models_All/cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_20.pth --data_path /home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542800988948006.pcd.bin
```
![2](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/d7a8b896-985c-42b3-8778-f161f3744e51)

# 三、按真值进行上色
修改tools/demo2.py中
```python
V.draw_scenes( # 绘制检测框
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=gt_labels,
                    point_colors=np.ones((data_dict['points'][:, 1:].shape[0], 3)), draw_origin=True
                )
```
为
```python
get_gt_boxes = True
            if get_gt_boxes:
                gt_boxes, gt_labels = get_gt() # 以samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin为例
                V.draw_scenes( # 绘制检测框
                points=data_dict['points'][:, 1:], ref_boxes=gt_boxes,
                ref_scores=np.ones_like(gt_labels) * 1, ref_labels=pred_dicts[0]['pred_labels'],
                point_colors=np.ones((data_dict['points'][:, 1:].shape[0], 3)), draw_origin=True)
            else:
                # points[0] = [x,y,z,intensity]
                V.draw_scenes( # 绘制检测框
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=gt_labels,
                    point_colors=np.ones((data_dict['points'][:, 1:].shape[0], 3)), draw_origin=True
                )
```
并添加get_gt()函数
```python
def get_gt():
    with open("/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl", 'rb') as f:
    # with open("/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/output/false_results/false_results.pkl", 'rb') as f:
        
        # 反序列化解析成列表a
        a = pickle.load(f)
        gt_boxes,gt_labels = [],[]
        name_label = {'car':0,'pedestrian':1}
        for i in a:
            if i['lidar_path'] == 'samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin':
                for j in range(len(i['gt_boxes'])):
                    gt_box = i['gt_boxes'][j]
                    try:
                        gt_label = name_label[i['gt_names'][j]]
                    except:
                        gt_label = 2
                    gt_boxes.append(gt_box)
                    gt_labels.append(gt_label)
                return gt_boxes, gt_labels
    return 0,0
```
可视化结果
```
python demo2.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ckpt /home/xingchen/Study/4D_GT/VoxelNeXt/output/nuscenes_models_All/cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_20.pth --data_path /home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin
```
![1](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/3e3f7c26-646d-4dca-8569-dd40e4be31f4)
![2](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/316b8d5e-3bf1-4f75-9253-dd636af9c537)
![3](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/f32b48b0-2ca1-4c9d-8887-e5b735303d38)
![4](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/a41bc88c-a3d8-4384-a6d0-8a658ab7d068)

# 四、地面上色
在tools/visual_utils/open3d_vis_utils.py中，直接按z坐标划分范围
```python
color = np.ones_like(points) * [1, 1, 1] # 初始颜色为白色
```
后添加
```python
color = add_color_ground(points,color) # 地面上色
```
并添加add_color_ground(points,color)函数
```python
def add_color_ground(points,color):
    color_on_the_ground = np.array([1, 1, 0])
    for i in range(len(points)):
        z = points[i][2]
        if z > -3 and z < -1:
            color[i] = color_on_the_ground
    return color
```
可视化结果，直接按z坐标划分范围效果较差
![5](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/b1c423fc-19d5-4949-b31b-c97d57848a15)

# 五、栅格可视化
每个点扩充成栅格
在tools/visual_utils/open3d_vis_utils.py中添加
```
if True: # 是否栅格化
        for point in points[:100]: # 取所有点的计算量太大，故取100个点做示例
            x,y,z = point[:3]
            size = 0.1
            bbox_lengths = np.array([size,size,size])  # 边界框的尺寸
            bbox_min_point = np.array([x-size/2,y-size/2,z-size/2])  # 边界框最左后下角点坐标

            # 创建边界框的立方体网格
            bbox_mesh = o3d.geometry.TriangleMesh.create_box(*bbox_lengths)
            bbox_mesh.compute_vertex_normals()

            # # 平移和缩放边界框网格
            bbox_mesh.translate(bbox_min_point)
            # bbox_mesh.scale(2 * bbox_lengths, center=(0, 0, 0))

            # 为边界框网格上色
            bbox_color = [1.0, 0.0, 0.0]  # 边界框的颜色
            bbox_mesh.paint_uniform_color(bbox_color)

            try:
                bbox_meshs= bbox_mesh + bbox_meshs
            except:
                bbox_meshs= bbox_mesh
        vis.add_geometry(bbox_meshs)
```
可视化结果
![2024-01-11 15-42-55屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/4ec96fcf-ebf5-459e-9d37-91473f5e7d73)
优化后
```python
if True: # 是否栅格化
        points = sorted(points, key=lambda p: (p[0], p[1], p[2])) # 排序
        count = 1
        for point in points[1000]:
            x,y,z = point[:3]
            size = 0.2
            bbox_lengths = np.array([size,size,size])  # 边界框的尺寸
            bbox_min_point = np.array([x-size/2,y-size/2,z-size/2])  # 边界框最左后下角点坐标

            # 创建边界框的立方体网格
            bbox_mesh = o3d.geometry.TriangleMesh.create_box(*bbox_lengths)
            bbox_mesh.compute_vertex_normals()

            # # 平移和缩放边界框网格
            bbox_mesh.translate(bbox_min_point)
            # bbox_mesh.scale(2 * bbox_lengths, center=(0, 0, 0))

            # 为边界框网格上色
            bbox_color = [1.0, 0.0, 0.0]  # 边界框的颜色
            bbox_mesh.paint_uniform_color(bbox_color)
            try:
                bbox_vertices = np.asarray(bbox_mesh.vertices)
                for i in bbox_vertices:
                    is_inside = Is_inside(bbox_mesh_old,i)
                    if is_inside:
                        break

                if is_inside:
                    continue
                else:
                    bbox_meshs= bbox_mesh + bbox_meshs
                    bbox_mesh_old = bbox_mesh
                    count = count+1
            except:
                bbox_meshs= bbox_mesh
                bbox_mesh_old = bbox_mesh
        vis.add_geometry(bbox_meshs)
        print(count)
```
并添加函数Is_inside(bbox_mesh,point)
```python
def Is_inside(bbox_mesh,point):
    # 获取边界框的顶点坐标
    bbox_vertices = np.asarray(bbox_mesh.vertices)

    # 获取边界框的最小和最大坐标
    bbox_min = np.min(bbox_vertices, axis=0)
    bbox_max = np.max(bbox_vertices, axis=0)

    # 判断点是否在边界框内部
    is_inside = np.all(point >= bbox_min) and np.all(point <= bbox_max)

    return is_inside
```
可视化结果
<table>
    <tr>
            <th>size = 0.1</th>
            <th>size = 0.2</th>
    </tr>
    <tr>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/f47cb7c3-ae66-4802-b40d-d453f2df6f68 /></td>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/71ee326c-5b8c-43a9-9855-f7452c8a0a40 /></td>
    </tr>
    <tr>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/ff46467f-c5fd-4d60-9bfa-bc25bb145578 /></td>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/8b85dde1-5714-4842-b50b-6c16392d9a9a /></td>
    </tr>
</table>
