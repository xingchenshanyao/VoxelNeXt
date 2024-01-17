# Occupancy
将栅格化结果按要求生成可视化视频与导出

## 一、另存现有程序
将tools/demo2_rasterization.py与tools/visual_utils/open3d_vis_utils.py分别另存为

tools/demo2_occupancy.py与tools/visual_utils/open3d_vis_utils_occupancy.py

把地面点、背景点的颜色分别改成橙色、黄色，可视化结果

<table>
    <tr>
            <th>size = 0.1</th>
            <th>size = 0.3</th>
    </tr>
    <tr>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/ee87dd7b-a09b-4c14-8e37-2837f423160b /></td>
        <td><img src=https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/5f14c2ac-1fbb-4b66-bd25-2d4a76607bf9 /></td>
    </tr>
</table>

## 二、固定相机视角
尝试使用固定视角代码失败
```python
    ctr = vis.get_view_control()
    ctr.set_lookat(np.array([0, 0, 0]))
    ctr.set_up((-1, 1, 1))  # 指向屏幕上方的向量
    ctr.set_front((1, -1, 1))  # 垂直指向屏幕外的向量
    ctr.set_zoom(0.15) # 控制远近
    vis.update_geometry(pts)
    vis.poll_events()
    vis.update_renderer()
```
经反复查询后，得知open3d==0.17.0存在BUG(kusi)，需要退回0.16.0
```
pip install open3d==0.16.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
可视化10帧后，拼接视频
```python
import os
import cv2
from PIL import Image


def image_to_video(image_path, media_path):
    '''
    图片合成视频函数
    :param image_path: 图片路径
    :param media_path: 合成视频保存路径
    :return:
    '''
    # 获取图片路径下面的所有图片名称
    image_names = os.listdir(image_path)
    # 对提取到的图片名称进行排序
    image_names.sort(key=lambda n: int(n[:-4]))
    # 设置写入格式
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    # 设置每秒帧数
    fps = 2  # 由于图片数目较少，这里设置的帧数比较低
    # 读取第一个图片获取大小尺寸，因为需要转换成视频的图片大小尺寸是一样的
    image = Image.open(image_path + image_names[0])
    # 初始化媒体写入对象
    media_writer = cv2.VideoWriter(media_path, fourcc, fps, image.size)
    # 遍历图片，将每张图片加入视频当中
    for image_name in image_names:
        im = cv2.imread(os.path.join(image_path, image_name))
        media_writer.write(im)
        print(image_name, '合并完成！')
    # 释放媒体写入对象
    media_writer.release()
    print('无声视频写入完成！')

# 图片路径
image_path =  "scene01/png/"
# 视频保存路径+名称
media_path = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/scene01/mp4/res1.mp4"
# 调用函数，生成视频
image_to_video(image_path, media_path)
```
可视化视频如下

https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/7a891a22-10f4-47f7-a6b4-78d78c60e7e8

## 三、重新可视化需求
- 将前后10帧的点云中，动态点去除，静态点在速度补偿后叠加
- 叠加后的点云数据与一组动态点叠加
- 进行栅格化
```python
import pickle
import open3d as o3d
import numpy as np
import os

def main():
    # Load PCD file
    # 地面点
    pcd_path1 = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/ground_output/1641634400.939636.pcd"
    # 背景点
    pcd_path2 = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/other_output/1641634400.939636.pcd"
    # 动态点
    pcd_path3 = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/move_output/1641634400.939636.pcd"
    
    point_cloud1 = o3d.io.read_point_cloud(pcd_path1)
    point_cloud2 = o3d.io.read_point_cloud(pcd_path2)
    # point_cloud3 = o3d.io.read_point_cloud(pcd_path3)
    
    points1 = np.asarray(point_cloud1.points)
    points2 = np.asarray(point_cloud2.points)
    # points3 = np.asarray(point_cloud3.points)

    points3 = move_points() # 从pkl文件中加载动态点
    point_cloud3 = o3d.geometry.PointCloud()
    point_cloud3.points = o3d.utility.Vector3dVector(points3)

    # # 获取点云的数量
    # num_points = len(points1)

    ## 体素的颜色取决于点云的颜色
    color1 = [1, 0.5, 0] # 地面点橙色
    point_cloud1.paint_uniform_color(color1)
    color2 = [1, 1, 0] # 背景点黄色
    point_cloud2.paint_uniform_color(color2)
    color3 = [1, 0, 0] # 动态点红色
    point_cloud3.paint_uniform_color(color3)

    # 合并
    points1 = point_cloud1.points
    colors1 = point_cloud1.colors
    points2 = point_cloud2.points
    colors2 = point_cloud2.colors
    points3 = point_cloud3.points
    colors3 = point_cloud3.colors


    merged_points = np.concatenate((points1, points2, points3))
    merged_colors = np.concatenate((colors1, colors2, colors3))
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(merged_points)
    points.colors = o3d.utility.Vector3dVector(merged_colors)

    # points = o3d.geometry.PointCloud.concatenate([point_cloud1, point_cloud2])
    # Gridify point cloud
    grid_resolution = 0.3
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(points, voxel_size=grid_resolution)

    print(f"Number of voxels: {len(voxel_grid.get_voxels())}")

    # 创建可视化窗口并添加点云
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(voxel_grid)

    # 获取渲染器控制对象
    render_option = vis.get_render_option()
    render_option.point_size = 0.01
    # 设置背景颜色为黑色
    render_option.background_color = np.asarray([0, 0, 0])

    ctr = vis.get_view_control()
    ctr.set_lookat(np.array([0, 0, 0]))
    ctr.set_up((-1, 1, 1))  # 指向屏幕上方的向量
    ctr.set_front((1, -1, 1))  # 垂直指向屏幕外的向量
    ctr.set_zoom(0.1) # 控制远近
    vis.update_geometry(voxel_grid)
    vis.poll_events()
    vis.update_renderer()

    vis.run()
    vis.destroy_window()

    # ply_path = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/voxel_grid.ply"
    # o3d.io.write_voxel_grid(ply_path, voxel_grid)

    # o3d.visualization.draw_geometries([voxel_grid])
    # o3d.visualization.draw_geometries([point_cloud, voxel_grid])

if __name__ == "__main__":
    main()
```
可视化结果
![1](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/c6b2b4e2-ee80-47ae-804a-43065a86e612)

