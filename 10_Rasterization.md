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
    color_in_bbox = box_colormap[label]
    # if label == 1:
    #     color_in_bbox = np.array([1, 0, 0])  # 红色
    # else:
    #     color_in_bbox = np.array([0, 1, 0])  # 绿色
    color = color_points_in_bbox(points, bbox_center, bbox_size, bbox_rotation, color_in_bbox,color)
    return color
```
