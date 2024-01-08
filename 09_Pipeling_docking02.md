# 09_Pipeline_docking02
将3D bbox与图像2D bbox匹配以确定bbox类别

参考来源：

https://blog.csdn.net/qq_45779334/article/details/126942387

## 一、NuScenes GT点云3D bbox投影到图像上
以nuscenes-devkit/python-sdk/nuscenes/scripts/export_2d_annotations_as_json.py为基础修改
```python
# nuScenes dev-kit.
# Code written by Sergi Adipraja Widjaja, 2019.

"""
Export 2D annotations (xmin, ymin, xmax, ymax) from re-projections of our annotated 3D bounding boxes to a .json file.

Note: Projecting tight 3d boxes to 2d generally leads to non-tight boxes.
      Furthermore it is non-trivial to determine whether a box falls into the image, rather than behind or around it.
      Finally some of the objects may be occluded by other objects, in particular when the lidar can see them, but the
      cameras cannot.
"""

import argparse
import json
import os
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

import cv2
import numpy as np

def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict,
                    x1: float,
                    y1: float,
                    x2: float,
                    y2: float,
                    sample_data_token: str,
                    filename: str) -> OrderedDict:
    """
    Generate one 2D annotation record given various informations on top of the 2D bounding box coordinates.
    :param ann_rec: Original 3d annotation record.
    :param x1: Minimum value of the x coordinate.
    :param y1: Minimum value of the y coordinate.
    :param x2: Maximum value of the x coordinate.
    :param y2: Maximum value of the y coordinate.
    :param sample_data_token: Sample data token.
    :param filename:The corresponding image file where the annotation is present.
    :return: A sample 2D annotation record.
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    return repro_rec


def get_2d_boxes(sample_data_token: str, visibilities: List[str]) -> List[OrderedDict]:
    """
    Get the 2D annotation records for a given `sample_data_token`.
    :param sample_data_token: Sample data token belonging to a camera keyframe.
    :param visibilities: Visibility filter.
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """
    # sample_data_token = 'e3d495d4ac534d54b321f50006683844'
    

    # Get the sample data and the sample corresponding to that sample data. # 获取与该样本数据对应的样本数据和样本
    sd_rec = nusc.get('sample_data', sample_data_token)
    # sd_rec = 
    # {
    # "token": "e3d495d4ac534d54b321f50006683844",
    # "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    # "ego_pose_token": "e3d495d4ac534d54b321f50006683844",
    # "calibrated_sensor_token": "1d31c729b073425e8e0202c5c6e66ee1",
    # "timestamp": 1532402927612460,
    # "fileformat": "jpg",
    # "is_key_frame": true,
    # "height": 900,
    # "width": 1600,
    # "filename": "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",
    # "prev": "",
    # "next": "68e8e98cf7b0487baa139df808641db7",
    # "sensor_modality": "camera",
    # "channel": "CAM_FRONT"
    # }

    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError('The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])
    # s_rec = 
    # {
    # "token": "ca9a282c9e77460f8360f564131a8af5",
    # "timestamp": 1532402927647951,
    # "prev": "",
    # "next": "39586f9d59004284a7114a68825e8eec",
    # "scene_token": "cc8c0bf57f984915a77078b10eb33198"
    # "data": {
    #         "RADAR_FRONT": "37091c75b9704e0daa829ba56dfa0906", ..., # 5个RADAR数据的索引
    #         "LIDAR_TOP": "9d9bf11fb0e144c8b446d54a8a00184f", 
    #         "CAM_FRONT": "e3d495d4ac534d54b321f50006683844", ... # 6个CAM数据的索引
    #         }
    # "anns": {
    #         'ef63a697930c4b20a6b9791f423351da', # 该图像的69个bbox的索引
    #         '6b89da9bf1f84fd6a5fbe1c3b236f809',
    #         ...
    #         }
    # }

    # Get the calibrated sensor and ego pose record to get the transformation matrices. 
    # 获取传感器变换矩阵
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    # cs_rec =
    # {
    # "token": "1d31c729b073425e8e0202c5c6e66ee1",
    # "sensor_token": "725903f5b62f56118f4094b46a4470d8",
    # "translation": [1.70079118954, 0.0159456324149, 1.51095763913],
    # "rotation": [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755],
    # "camera_intrinsic": 
    #                     [
    #                     [1266.417203046554, 0.0, 816.2670197447984],
    #                     [0.0, 1266.417203046554, 491.50706579294757],
    #                     [0.0, 0.0, 1.0]
    #                     ]
    # }

    # 获取车身位置变换矩阵
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    # pose_rec =
    # {
    # "token": "e3d495d4ac534d54b321f50006683844",
    # "timestamp": 1532402927612460,
    # "rotation": [0.5720063498929273, -0.0021434844534272707, 0.011564094980151613, -0.8201648693182716],
    # "translation": [411.4199861830012, 1181.197175631848, 0.0]
    # }
    
    # 获取相机内参
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])
    # camera_intrinsic =
    # [
    # [1266.417203046554, 0.0, 816.2670197447984],
    # [0.0, 1266.417203046554, 491.50706579294757],
    # [0.0, 0.0, 1.0]
    # ]

    # Get all the annotation with the specified visibilties. 
    # 获取具有指定可见性的所有注释
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in visibilities)]
    # ann_recs[0] =  # ann_recs为这张图片内的69个检测框的信息
    # {
    # "token": "ef63a697930c4b20a6b9791f423351da",
    # "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    # "instance_token": "6dd2cbf4c24b4caeb625035869bca7b5",
    # "visibility_token": "1",
    # "attribute_tokens": ["4d8821270b4a47e3a8a300cbec48188e"],
    # "translation": [373.256, 1130.419, 0.8],
    # "size": [0.621, 0.669, 1.642],
    # "rotation": [0.9831098797903927, 0.0, 0.0, -0.18301629506281616],
    # "prev": "",
    # "next": "7987617983634b119e383d8a29607fd7",
    # "num_lidar_pts": 1,
    # "num_radar_pts": 0,
    # "category_name": "human.pedestrian.adult"
    # }

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        # ann_rec['sample_annotation_token'] = 'ef63a697930c4b20a6b9791f423351da'

        ann_rec['sample_data_token'] = sample_data_token
        # ann_rec['sample_data_token'] = 'e3d495d4ac534d54b321f50006683844'

        # Get the box in global coordinates.
        # 获取在 世界坐标系 下的bbox坐标
        box = nusc.get_box(ann_rec['token'])
        # box = 
        # {
        # label: nan, 
        # score: nan, 
        # xyz: [373.26, 1130.42, 0.80], 
        # wlh: [0.62, 0.67, 1.64], 
        # rot axis: [0.00, 0.00, -1.00], 
        # ang(degrees): 21.09, 
        # ang(rad): 0.37, 
        # vel: nan, nan, nan, 
        # name: human.pedestrian.adult, 
        # token: ef63a697930c4b20a6b9791f423351da
        # }

        # Move them to the ego-pose frame.
        # 把 世界坐标系 下坐标变成 Ego自身坐标系 下坐标
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)
        # box = 
        # {
        # label: nan, 
        # score: nan, 
        # xyz: [60.83, -18.29, 1.00], 
        # wlh: [0.62, 0.67, 1.64], 
        # rot axis: [0.01, -0.02, 1.00], 
        # ang(degrees): 89.13, 
        # ang(rad): 1.56, 
        # vel: nan, nan, nan, 
        # name: human.pedestrian.adult, 
        # token: ef63a697930c4b20a6b9791f423351da
        # }


        # Move them to the calibrated sensor frame.
        # 把 Ego自身坐标系 下坐标变成 相机坐标系 下坐标
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)
        # box =
        # {
        # label: nan, 
        # score: nan, 
        # xyz: [18.64, 0.19, 59.02], 
        # wlh: [0.62, 0.67, 1.64], 
        # rot axis: [0.02, -0.71, 0.70], 
        # ang(degrees): -179.94, 
        # ang(rad): -3.14, 
        # vel: nan, nan, nan, 
        # name: human.pedestrian.adult, 
        # token: ef63a697930c4b20a6b9791f423351da
        # }

        # Filter out the corners that are not in front of the calibrated sensor.
        # 过滤掉不在校准传感器前方的bbox。
        corners_3d = box.corners()
        # 计算出bbox8个角点的空间位置坐标
        # corners_3d = array(
        # [[18.31613738, 18.32923937, 18.29283561, 18.27973361, 18.98482399, 18.99792598, 18.96152222, 18.94842023],       
        #  [-0.62997011, -0.63928954,  1.00211021,  1.01142964, -0.61492468, -0.62424411,  1.01715564,  1.02647507],       
        #  [58.70871026, 59.32950208, 59.35491135, 58.73411953, 58.69482329, 59.31561511, 59.34102439, 58.72023256]])
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        # 返回在当前CAM前方的角点序号
        # in_front = array([0, 1, 2, 3, 4, 5, 6, 7])
        corners_3d = corners_3d[:, in_front]
        # 获取在当前CAM前方的角点

        # Project 3d box to 2d.
        # 3d bbox转成2d bbox
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
        # 8个角点投影到图像后的坐标
        # corner_coords = 
        # [
        # [1211.3680424781949, 477.9178558746593], 
        # [1207.5135875496399, 477.86111828160216], 
        # [1206.5693751819117, 512.8884406232007], 
        # [1210.412182737279, 513.3153758758543], 
        # [1225.8893058022245, 478.2392654711055], 
        # [1221.8819702506732, 478.1791507511069], 
        # [1220.9313845115814, 513.2145339950961], 
        # [1224.9269364018471, 513.6450176828284]
        # ]

        # Keep only corners that fall within the image.
        # 只保留图像内的角点
        final_coords = post_process_coords(corner_coords)
        # final_coords = (1206.5693751819117, 477.86111828160216, 1225.8893058022245, 513.6450176828284)
        # min_x, min_y, max_x, max_y

        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        if (max_x - min_x)  > 200:
            cam_rec = nusc.get('sample_data', s_rec['data']['CAM_FRONT'])
            image = os.path.join('/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/data/nuscenes/v1.0-mini', \
                                    cam_rec['filename'])
            if False:
                draw_3d_bbox(image,corner_coords)
            if True:
                draw_2d_bbox(image,final_coords)

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec['filename'])
        repro_recs.append(repro_rec)

    return repro_recs


def main(args):
    """Generates 2D re-projections of the 3D bounding boxes present in the dataset."""

    print("Generating 2D reprojections of the nuScenes dataset")

    # Get tokens for all camera images. # 加载所有6个相机图片文件夹中2424张图片的索引
    sample_data_camera_tokens = [s['token'] for s in nusc.sample_data if (s['sensor_modality'] == 'camera') and
                                 s['is_key_frame']]
    # 扫描sample_data.json文件中取出每一项，例如，
    # {
    # "token": "5ace90b379af485b9dcb1584b01e7212",
    # "sample_token": "39586f9d59004284a7114a68825e8eec",
    # "ego_pose_token": "5ace90b379af485b9dcb1584b01e7212",
    # "calibrated_sensor_token": "f4d2a6c281f34a7eb8bb033d82321f79",
    # "timestamp": 1532402927814384,
    # "fileformat": "pcd",
    # "is_key_frame": false,
    # "height": 0,
    # "width": 0,
    # "filename": "sweeps/RADAR_FRONT/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402927814384.pcd",
    # "prev": "f0b8593e08594a3eb1152c138b312813",
    # "next": "978db2bcdf584b799c13594a348576d2"
    # }
    # 'sensor_modality': 'radar' # 这两项应该是通过filename判断提取的
    # 'channel': 'RADAR_FRONT'
    # 保留其中'sensor_modality'=='camera'和"is_key_frame" == true的项的token
    # sample_data_camera_tokens = ['e3d495d4ac534d54b321f50006683844', ...] # 共2424项



    # For debugging purposes: Only produce the first n images. # 调试时使用
    if True: # 20240106
        sample_data_camera_tokens = sample_data_camera_tokens[:1]
    if args.image_limit != -1:
        sample_data_camera_tokens = sample_data_camera_tokens[:args.image_limit]

    # Loop through the records and apply the re-projection algorithm. # 循环浏览记录并应用重新投影算法
    reprojections = []
    for token in tqdm(sample_data_camera_tokens):
        reprojection_records = get_2d_boxes(token, args.visibilities)
        reprojections.extend(reprojection_records)

    # Save to a .json file.
    dest_path = os.path.join(args.dataroot, args.version)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with open(os.path.join(args.dataroot, args.version, args.filename), 'w') as fh:
        json.dump(reprojections, fh, sort_keys=True, indent=4)

    print("Saved the 2D re-projections under {}".format(os.path.join(args.dataroot, args.version, args.filename)))

def draw_3d_bbox(image,corner_coords):
    points = [(int(corner_coords[0][0]), int(corner_coords[0][1])), 
              (int(corner_coords[1][0]), int(corner_coords[1][1])), 
              (int(corner_coords[2][0]), int(corner_coords[2][1])), 
              (int(corner_coords[3][0]), int(corner_coords[3][1])), 
              (int(corner_coords[4][0]), int(corner_coords[4][1])), 
              (int(corner_coords[5][0]), int(corner_coords[5][1])), 
              (int(corner_coords[6][0]), int(corner_coords[6][1])), 
              (int(corner_coords[7][0]), int(corner_coords[7][1]))]

    img = cv2.imread(image)
    # 将边界框的点连接起来
    cv2.line(img, points[0], points[1], (0, 255, 0), 1)
    cv2.line(img, points[1], points[2], (0, 255, 0), 1)
    cv2.line(img, points[2], points[3], (0, 255, 0), 1)
    cv2.line(img, points[3], points[0], (0, 255, 0), 1)
    cv2.line(img, points[4], points[5], (0, 255, 0), 1)
    cv2.line(img, points[5], points[6], (0, 255, 0), 1)
    cv2.line(img, points[6], points[7], (0, 255, 0), 1)
    cv2.line(img, points[7], points[4], (0, 255, 0), 1)
    cv2.line(img, points[0], points[4], (0, 255, 0), 1)
    cv2.line(img, points[5], points[1], (0, 255, 0), 1)
    cv2.line(img, points[6], points[2], (0, 255, 0), 1)
    cv2.line(img, points[7], points[3], (0, 255, 0), 1)

    # 显示图像
    cv2.imshow("Bounding Box", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

def draw_2d_bbox(image,final_coords):
    min_x, min_y, max_x, max_y = list(final_coords)
    points = [(int(min_x),int(max_y)),
              (int(max_x),int(max_y)),
              (int(max_x),int(min_y)),
              (int(min_x),int(min_y))]

    img = cv2.imread(image)
    # 将边界框的点连接起来
    cv2.line(img, points[0], points[1], (0, 255, 0), 1)
    cv2.line(img, points[1], points[2], (0, 255, 0), 1)
    cv2.line(img, points[2], points[3], (0, 255, 0), 1)
    cv2.line(img, points[3], points[0], (0, 255, 0), 1)

    # 显示图像
    cv2.imshow("Bounding Box", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export 2D annotations from reprojections to a .json file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/data/nuscenes/v1.0-mini', help="Path where nuScenes is saved.")
    parser.add_argument('--version', type=str, default='v1.0-mini', help='Dataset version.')
    parser.add_argument('--filename', type=str, default='2D-box00000.json', help='Output filename.')
    parser.add_argument('--visibilities', type=str, default=['', '1', '2', '3', '4'],
                        help='Visibility bins, the higher the number the higher the visibility.', nargs='+')
    parser.add_argument('--image_limit', type=int, default=-1, help='Number of images to process or -1 to process all.')
    args = parser.parse_args()

    nusc = NuScenes(dataroot=args.dataroot, version=args.version)
    main(args)
```
单个3D bbox投影可视化结果

![1](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/71c6a905-24af-413e-8f52-98fb1a69349b)

![2](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/c7f7c84b-a4bc-4c00-94e2-77a042570790)
