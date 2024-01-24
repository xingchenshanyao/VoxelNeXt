# Occupancy
将栅格化结果按要求生成可视化视频与导出栅格真值文件

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

## 四、更换新的栅格化方法
对原始数据集进行栅格化
```
输入：每一帧的地面点、背景点和动态(car)点 .pcd文件，文件名称均为lidar_timestamp.pcd，点云坐标系均为当前帧激光雷达坐标系。
输出：所有帧的栅格化信息的 .npy文件及可视化 .png文件，保存名称为lidar_timestamp.npy和lidar_timestamp.png。
处理过程：
step1:每一帧按顺序处理，加载当前帧的地面点、背景点和动态点，地面点和背景点按体素方法(voxel_size=0.3)做下采样，并适当减小点云空间范围到[[-18,25],[-10,80],[-3,3]]，删除范围外点，以减少后续计算量。
step2:地面点、背景点和动态点分别上色(地面点紫色[0.306, 0.114, 0.298]、背景点橙色[0.859, 0.455, 0.169]、动态点淡红色[0.820, 0.286, 0.306])。
step3:将三者合并，计算占用总点云空间大小，按cube_size=0.5生成n个栅格，遍历每一个栅格，判断合并后有无点在栅格内，没有则跳过该栅格，有则保留栅格并读取栅格内点的平均颜色作为栅格颜色，栅格内类别数量最多的点类别作为栅格类别c，最后保留栅格数量为m。
step4:将栅格化场景可视化并保存 .png文件，将栅格信息保存 .npy文件，.npy文件存储(m,4)的栅格信息，每一列为栅格中心点xyz坐标和栅格类别c。栅格尺寸统一为0.5m。
```
代码为main_occupany_self_visualize.py
```python
import pickle
import open3d as o3d
import numpy as np
import os

def draw_line(vertices):
    # vertices[2],vertices[3],vertices[6],vertices[7] = vertices[3],vertices[2],vertices[7],vertices[6]
    edges = np.array([
    [0, 1], [1, 3], [3, 2], [2, 0],  # 前面四条边
    [4, 5], [5, 7], [7, 6], [6, 4],  # 后面四条边
    [0, 4], [1, 5], [2, 6], [3, 7]   # 连接前后的四条边
    ])

    # 创建LineSet对象
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(vertices)
    lines.lines = o3d.utility.Vector2iVector(edges)
    line_color = [1, 1, 1]  
    lines.colors = o3d.utility.Vector3dVector([line_color] * 8)
    return lines



def No_points(vertices,point_cloud,all_in_points):
    points = point_cloud.points
    colors = point_cloud.colors
    points0 = sorted(points, key=lambda p: (p[0], p[1], p[2]))

    cube_min, cube_max = vertices[0], vertices[-1]
    
    in_points = []

    for j in range(0,len(points),1): # 理想情况不应该有10这个间隔，但算力不足
        if j in all_in_points: # all_in_points 目的是减少计算量
            continue
        point = points[j]
        if np.all(cube_min <= point) and np.all(point <= cube_max):
            in_points.append(colors[j])
            all_in_points.append(j)
            if len(in_points) > 10: # 减小计算量
                break
    if len(in_points) == 0:
        return None,all_in_points
    cube_color = np.mean(in_points, axis=0)
    return cube_color.tolist(),all_in_points


def transformation(points_nx3,tran):
    n = len(points_nx3)
    ones_column = np.ones((n, 1))
    points_nx4 = np.hstack((points_nx3, ones_column))
    tran_inv = np.linalg.inv(tran)
    points_4xn = np.dot(tran_inv, points_nx4.T)
    points_tran = points_4xn.T[:,:3]
    return points_tran



def move_points(time_name):
    pkl_dir = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/move_output/pkl_track"
    names = os.listdir(pkl_dir)
    for name in names:
        pkl_path = pkl_dir + '/' + name
        with open(pkl_path, 'rb') as f:        
            # 反序列化解析成列表a
            f = pickle.load(f)
            points1 = None
            for i in f:
                if i['fn'] == str(time_name):
                    points1 = i['pcd_body'][:,:3]
                    try:
                        points = np.concatenate((points,points1), axis=0)
                    except:
                        points = points1
                    break
    try:
        points_swapped = [[y, -x, z] for x, y, z in points]
        # print(points_swapped[0])
    except:
        points_swapped = None
    return points_swapped

def draw(point_cloud,cubes,cube_linesets,save_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    # vis.add_geometry(bounding_box)
    for i in range(len(cubes)):
        vis.add_geometry(cubes[i])
        vis.add_geometry(cube_linesets[i])

    # 设置渲染选项
    render_options = vis.get_render_option()
    render_options.point_size = 1
    render_options.light_on = False # 关闭光渲染
    ctr = vis.get_view_control()
    ctr.set_lookat(np.array([0, -10, 0]))

    ctr.set_up((0, -1, 3))  # 指向屏幕上方的向量
    ctr.set_front((0, 3, 1))  # 垂直指向屏幕外的向量
    ctr.set_zoom(0.06) # 控制远近
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path) # 保存当前画面
    print(save_path,"is OK !")
    # 运行可视化窗口
    # vis.run()

    # 关闭可视化窗口
    vis.destroy_window()

# 先下采样减小计算量，划分栅格，判断栅格内有无点，剔除无点的栅格，缺点是计算量过于庞大
def main():
    save_dir = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/jpg_output3/"
    # 地面点
    pcd_dir1 = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/ground_output/"
    # 背景点
    pcd_dir2 = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/other_output/"

    txt_path = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/lidar_pose_4f.txt"
    names = os.listdir(pcd_dir1)
    names = sorted(names)
    num = -1 # 记录偏移系数
    all_num = len(names)
    for name in names:
        num = num+1
        # name = names[70]
        # name = "1641634400.939636.pcd"
        # name = names[10]
        time_name = name[:-4]

        with open(txt_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                line.strip()

                line_list = line.split(',')
                if line_list[0] == time_name:
                    l = line_list
                    transformation_matrix = np.array([
                        [l[1], l[2], l[3], l[4]],
                        [l[5], l[6], l[7], l[8]],
                        [l[9], l[10], l[11], l[12].split('\n')[0]],
                        [0,   0,   0,   1]
                    ])
                    transformation_matrix = transformation_matrix.astype(float)
                    break

        pcd_path1 = pcd_dir1 + name # 地面点
        pcd_path2 = pcd_dir2 + name # 背景点
        save_path = save_dir + time_name + '.png'
        
        
        point_cloud1 = o3d.io.read_point_cloud(pcd_path1)
        point_cloud2 = o3d.io.read_point_cloud(pcd_path2)


        # point_cloud3 = o3d.io.read_point_cloud(pcd_path3)
        
        # 下采样
        point_cloud1 = point_cloud1.voxel_down_sample(voxel_size=0.3) 
        point_cloud2 = point_cloud2.voxel_down_sample(voxel_size=0.3)

        # 去点部分点:地面点和背景点中只保留 
        # -51.2<x<51.2 -51.2<y<51.2 -5<z<3
        # range_xyz = [[-51.2,51.2],[-51.2,51.2],[-5,3]]
        # -80<x<10 -20<y<20 -1<z<3
        range_xyz = [[-18,25],[-80,10],[-3,3]]


        points1 = np.asarray(point_cloud1.points)
        points1 = points1[(points1[:,0] < range_xyz[0][1])&(points1[:,0] > range_xyz[0][0])]
        points1 = points1[(points1[:,1] < range_xyz[1][1])&(points1[:,1] > range_xyz[1][0])]
        points1 = points1[(points1[:,2] < range_xyz[1][1])&(points1[:,2] > range_xyz[2][0])]
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud1.points = o3d.utility.Vector3dVector(points1)

        points2 = np.asarray(point_cloud2.points)
        points2 = points2[(points2[:,0] < range_xyz[0][1])&(points2[:,0] > range_xyz[0][0])]
        points2 = points2[(points2[:,1] < range_xyz[1][1])&(points2[:,1] > range_xyz[1][0])]
        points2 = points2[(points2[:,2] < range_xyz[2][1])&(points2[:,2] > range_xyz[2][0])]
        point_cloud2 = o3d.geometry.PointCloud()
        point_cloud2.points = o3d.utility.Vector3dVector(points2)



        points1 = np.asarray(point_cloud1.points)
        points2 = np.asarray(point_cloud2.points)
        print("len(points1) = ", len(points1))
        print("len(points2) = ", len(points2))

        # points1, points2 = transformation(points1,transformation_matrix),transformation(points2,transformation_matrix)
        # sorted_indices1 = np.argsort(points1[:, 0])
        # points1 = points1[sorted_indices1]
        # sorted_indices2 = np.argsort(points2[:, 0])
        # points2 = points2[sorted_indices2]

        ## 体素的颜色取决于点云的颜色
        # color1 = [0, 0, 0] # 地面点黑色
        # color1 = [0.243, 0.737, 0.792] # 地面点蓝色
        color1 = [0.306, 0.114, 0.298] # 地面点紫色 78 29 76
        point_cloud1.paint_uniform_color(color1)
        # color2 = [0.725, 0.702, 0.651] # 背景点灰色
        # color2 = [0.204, 0.455, 0.251] # 背景点绿色
        color2 = [0.859, 0.455, 0.169] # 背景点橙色 219 116 43
        point_cloud2.paint_uniform_color(color2)
        

        # 合并
        # points1 = point_cloud1.points
        colors1 = point_cloud1.colors
        # points2 = point_cloud2.points
        colors2 = point_cloud2.colors


        points3 = None
        points3 = move_points(time_name) # 从pkl文件中加载动态点
        if points3 is not None:
            # sorted_indices3 = np.argsort(points3[:, 0])
            # points3 = points3[sorted_indices3]
            point_cloud3 = o3d.geometry.PointCloud()
            point_cloud3.points = o3d.utility.Vector3dVector(points3)
            # color3 = [0.118, 0.231, 0.478] # 动态点撒蓝色 
            # color3 = [1, 0, 0] # 动态点红色 
            color3 = [0.820, 0.286, 0.306] # 动态点淡红色 209 73 78
            point_cloud3.paint_uniform_color(color3)

            # 下采样
            point_cloud3 = point_cloud3.voxel_down_sample(voxel_size=0.3)

            points3 = point_cloud3.points
            colors3 = point_cloud3.colors
            print("len(points3) = ", len(points3))
            merged_points = np.concatenate((points1, points2, points3))
            merged_colors = np.concatenate((colors1, colors2, colors3))
        else:
            merged_points = np.concatenate((points1, points2))
            merged_colors = np.concatenate((colors1, colors2))

        # 创建点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(merged_points)
        point_cloud.colors = o3d.utility.Vector3dVector(merged_colors)

        # 创建边界框几何体
        bounding_box = point_cloud.get_axis_aligned_bounding_box()
        bounding_box.color = (1, 0, 0)
        center = bounding_box.get_center()
        extent = bounding_box.get_extent()
        print("边界框中心点：", center)
        print("边界框长宽高：", extent)

        # 计算每个立方体框的边长为0.5
        cube_size = 0.5

        # 计算填充的立方体框数量
        num_cubes_x = int(np.ceil(extent[0] / cube_size))
        num_cubes_y = int(np.ceil(extent[1] / cube_size))
        num_cubes_z = int(np.ceil(extent[2] / cube_size))
        print("max_len(cubes) =",num_cubes_x*num_cubes_y*num_cubes_z)


        # 创建封闭的绿色立方体框
        cubes = []
        cube_linesets = []
        all_in_points = []
        for i in range(num_cubes_x):
            for j in range(num_cubes_y):
                for k in range(num_cubes_z):
                    cube_center = center +np.array([i, j, k]) * cube_size - 0.5*extent
                    cube = o3d.geometry.TriangleMesh.create_box(width=cube_size, height=cube_size, depth=cube_size)
                    cube.compute_vertex_normals()
                    cube.translate(cube_center)
                    vertices = np.asarray(cube.vertices)
                    # print(vertices)
                    # 判断这个cube内有没有点
                    range_xyz = [[vertices[0][0],vertices[-1][0]],[vertices[0][1],vertices[-1][1]],[vertices[0][2],vertices[-1][2]]]
                    points = np.asarray(point_cloud.points)
                    colors = np.asarray(point_cloud.colors)
                    index = (points[:,0] < range_xyz[0][1])&(points[:,0] > range_xyz[0][0])&(points[:,1] < range_xyz[1][1])&(points[:,1] > range_xyz[1][0])&(points[:,2] < range_xyz[2][1])&(points[:,2] > range_xyz[2][0])
                    points = points[index]

                    # points = points[(points[:,1] < range_xyz[1][1])&(points[:,1] > range_xyz[1][0])]
                    # points = points[(points[:,2] < range_xyz[2][1])&(points[:,2] > range_xyz[2][0])]

                    if len(points) == 0:
                        continue
                    else:
                        colors = colors[index]
                        color = np.mean(colors, axis=0)
                        cube.paint_uniform_color(color)


                    # color,all_in_points = No_points(vertices,point_cloud,all_in_points)
                    # if color is None:
                    #     continue
                    # else:
                    #     cube.paint_uniform_color(color)

                    # cube.paint_uniform_color([0.0, 1.0, 0.0])  # 设置颜色为绿色
                    cubes.append(cube)
                    lineset = draw_line(vertices)
                    cube_linesets.append(lineset)

                    if len(cubes) % 10 == 0:
                        print("len(cubes) =",len(cubes),", schedule =","%.3f"%(i/num_cubes_x),", all_schedule =","%.3f"%(num/all_num))
            #         break
            #     break
            # break

        print("len(final_cubes) =",len(cubes))

        draw(point_cloud,cubes,cube_linesets,save_path)
        # break

if __name__ == "__main__":
    main()
```
可视化效果
![1641634400 939636](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/ecc752cd-a219-4cc2-b225-bb0f890df339)

## 五、导出栅格化真值
代码为main_occupancy_self_output_npy.py
```python
import pickle
import open3d as o3d
import numpy as np
import os

def draw_line(vertices):
    # vertices[2],vertices[3],vertices[6],vertices[7] = vertices[3],vertices[2],vertices[7],vertices[6]
    edges = np.array([
    [0, 1], [1, 3], [3, 2], [2, 0],  # 前面四条边
    [4, 5], [5, 7], [7, 6], [6, 4],  # 后面四条边
    [0, 4], [1, 5], [2, 6], [3, 7]   # 连接前后的四条边
    ])

    # 创建LineSet对象
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(vertices)
    lines.lines = o3d.utility.Vector2iVector(edges)
    line_color = [1, 1, 1]  
    lines.colors = o3d.utility.Vector3dVector([line_color] * 8)
    return lines



def No_points(vertices,point_cloud,all_in_points):
    points = point_cloud.points
    colors = point_cloud.colors
    points0 = sorted(points, key=lambda p: (p[0], p[1], p[2]))

    cube_min, cube_max = vertices[0], vertices[-1]
    
    in_points = []

    for j in range(0,len(points),1): # 理想情况不应该有10这个间隔，但算力不足
        if j in all_in_points: # all_in_points 目的是减少计算量
            continue
        point = points[j]
        if np.all(cube_min <= point) and np.all(point <= cube_max):
            in_points.append(colors[j])
            all_in_points.append(j)
            if len(in_points) > 10: # 减小计算量
                break
    if len(in_points) == 0:
        return None,all_in_points
    cube_color = np.mean(in_points, axis=0)
    return cube_color.tolist(),all_in_points


def transformation(points_nx3,tran):
    n = len(points_nx3)
    ones_column = np.ones((n, 1))
    points_nx4 = np.hstack((points_nx3, ones_column))
    tran_inv = np.linalg.inv(tran)
    points_4xn = np.dot(tran_inv, points_nx4.T)
    points_tran = points_4xn.T[:,:3]
    return points_tran



def move_points(time_name):
    pkl_dir = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/move_output/pkl_track"
    names = os.listdir(pkl_dir)
    for name in names:
        pkl_path = pkl_dir + '/' + name
        with open(pkl_path, 'rb') as f:        
            # 反序列化解析成列表a
            f = pickle.load(f)
            points1 = None
            for i in f:
                if i['fn'] == str(time_name):
                    points1 = i['pcd_body'][:,:3]
                    try:
                        points = np.concatenate((points,points1), axis=0)
                    except:
                        points = points1
                    break
    try:
        points_swapped = [[y, -x, z] for x, y, z in points]
        # print(points_swapped[0])
    except:
        points_swapped = None
    return points_swapped

def draw(point_cloud,cubes,cube_linesets,save_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    # vis.add_geometry(bounding_box)
    for i in range(len(cubes)):
        vis.add_geometry(cubes[i])
        vis.add_geometry(cube_linesets[i])

    # 设置渲染选项
    render_options = vis.get_render_option()
    render_options.point_size = 1
    render_options.light_on = False # 关闭光渲染
    ctr = vis.get_view_control()
    ctr.set_lookat(np.array([0, -10, 0]))

    ctr.set_up((0, -1, 3))  # 指向屏幕上方的向量
    ctr.set_front((0, 3, 1))  # 垂直指向屏幕外的向量
    ctr.set_zoom(0.06) # 控制远近
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    # vis.capture_screen_image(save_path) # 保存当前画面
    # print(save_path,"is OK !")
    # 运行可视化窗口
    vis.run()

    # 关闭可视化窗口
    vis.destroy_window()

# 先下采样减小计算量，划分栅格，判断栅格内有无点，剔除无点的栅格，缺点是计算量过于庞大
def main():
    # save_dir = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/jpg_output3/"

    # 保存栅格文件
    save_dir = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/npy_output/"
    # 地面点
    pcd_dir1 = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/ground_output/"
    # 背景点
    pcd_dir2 = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/other_output/"

    txt_path = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/lidar_pose_4f.txt"
    names = os.listdir(pcd_dir1)
    names = sorted(names)
    num = -1 # 记录偏移系数
    all_num = len(names)
    for name in names:
        num = num+1
        # name = names[70]
        # name = "1641634400.939636.pcd"
        # name = names[10]
        time_name = name[:-4]

        with open(txt_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                line.strip()

                line_list = line.split(',')
                if line_list[0] == time_name:
                    l = line_list
                    transformation_matrix = np.array([
                        [l[1], l[2], l[3], l[4]],
                        [l[5], l[6], l[7], l[8]],
                        [l[9], l[10], l[11], l[12].split('\n')[0]],
                        [0,   0,   0,   1]
                    ])
                    transformation_matrix = transformation_matrix.astype(float)
                    break

        pcd_path1 = pcd_dir1 + name # 地面点
        pcd_path2 = pcd_dir2 + name # 背景点
        # save_path = save_dir + time_name + '.png'
        save_path = save_dir + time_name + '.npy'
        
        
        point_cloud1 = o3d.io.read_point_cloud(pcd_path1)
        point_cloud2 = o3d.io.read_point_cloud(pcd_path2)


        # point_cloud3 = o3d.io.read_point_cloud(pcd_path3)
        
        # 下采样
        point_cloud1 = point_cloud1.voxel_down_sample(voxel_size=0.3) 
        point_cloud2 = point_cloud2.voxel_down_sample(voxel_size=0.3)

        # 去点部分点:地面点和背景点中只保留 
        # -51.2<x<51.2 -51.2<y<51.2 -5<z<3
        # range_xyz = [[-51.2,51.2],[-51.2,51.2],[-5,3]]
        # -80<x<10 -20<y<20 -1<z<3
        range_xyz = [[-18,25],[-80,10],[-3,3]]


        points1 = np.asarray(point_cloud1.points)
        points1 = points1[(points1[:,0] < range_xyz[0][1])&(points1[:,0] > range_xyz[0][0])]
        points1 = points1[(points1[:,1] < range_xyz[1][1])&(points1[:,1] > range_xyz[1][0])]
        points1 = points1[(points1[:,2] < range_xyz[1][1])&(points1[:,2] > range_xyz[2][0])]
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud1.points = o3d.utility.Vector3dVector(points1)

        points2 = np.asarray(point_cloud2.points)
        points2 = points2[(points2[:,0] < range_xyz[0][1])&(points2[:,0] > range_xyz[0][0])]
        points2 = points2[(points2[:,1] < range_xyz[1][1])&(points2[:,1] > range_xyz[1][0])]
        points2 = points2[(points2[:,2] < range_xyz[2][1])&(points2[:,2] > range_xyz[2][0])]
        point_cloud2 = o3d.geometry.PointCloud()
        point_cloud2.points = o3d.utility.Vector3dVector(points2)



        points1 = np.asarray(point_cloud1.points)
        points2 = np.asarray(point_cloud2.points)
        print("len(points1) = ", len(points1))
        print("len(points2) = ", len(points2))

        # points1, points2 = transformation(points1,transformation_matrix),transformation(points2,transformation_matrix)
        # sorted_indices1 = np.argsort(points1[:, 0])
        # points1 = points1[sorted_indices1]
        # sorted_indices2 = np.argsort(points2[:, 0])
        # points2 = points2[sorted_indices2]

        ## 体素的颜色取决于点云的颜色
        color1 = [0.1, 0, 0] # 地面点黑色
        # color1 = [0.243, 0.737, 0.792] # 地面点蓝色
        # color1 = [0.306, 0.114, 0.298] # 地面点紫色 78 29 76
        point_cloud1.paint_uniform_color(color1)
        color2 = [0.2, 1, 0] # 背景点绿色
        # color2 = [0.725, 0.702, 0.651] # 背景点灰色
        # color2 = [0.204, 0.455, 0.251] # 背景点绿色
        # color2 = [0.859, 0.455, 0.169] # 背景点橙色 219 116 43
        point_cloud2.paint_uniform_color(color2)
        

        # 合并
        # points1 = point_cloud1.points
        colors1 = point_cloud1.colors
        # points2 = point_cloud2.points
        colors2 = point_cloud2.colors


        points3 = None
        points3 = move_points(time_name) # 从pkl文件中加载动态点
        if points3 is not None:
            # sorted_indices3 = np.argsort(points3[:, 0])
            # points3 = points3[sorted_indices3]
            point_cloud3 = o3d.geometry.PointCloud()
            point_cloud3.points = o3d.utility.Vector3dVector(points3)
            # color3 = [0.118, 0.231, 0.478] # 动态点撒蓝色 
            color3 = [1, 0, 0] # 动态点红色 
            # color3 = [0.820, 0.286, 0.306] # 动态点淡红色 209 73 78
            point_cloud3.paint_uniform_color(color3)

            # 下采样
            point_cloud3 = point_cloud3.voxel_down_sample(voxel_size=0.3)

            points3 = point_cloud3.points
            colors3 = point_cloud3.colors
            print("len(points3) = ", len(points3))
            merged_points = np.concatenate((points1, points2, points3))
            merged_colors = np.concatenate((colors1, colors2, colors3))
        else:
            merged_points = np.concatenate((points1, points2))
            merged_colors = np.concatenate((colors1, colors2))

        # 创建点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(merged_points)
        point_cloud.colors = o3d.utility.Vector3dVector(merged_colors)

        # 创建边界框几何体
        bounding_box = point_cloud.get_axis_aligned_bounding_box()
        bounding_box.color = (1, 0, 0)
        center = bounding_box.get_center()
        extent = bounding_box.get_extent()
        print("边界框中心点：", center)
        print("边界框长宽高：", extent)

        # 计算每个立方体框的边长为0.5
        cube_size = 0.5

        # 计算填充的立方体框数量
        num_cubes_x = int(np.ceil(extent[0] / cube_size))
        num_cubes_y = int(np.ceil(extent[1] / cube_size))
        num_cubes_z = int(np.ceil(extent[2] / cube_size))
        print("max_len(cubes) =",num_cubes_x*num_cubes_y*num_cubes_z)


        # 创建封闭的绿色立方体框
        cubes = []
        cube_linesets = []
        save_cubes = []
        for i in range(num_cubes_x):
            for j in range(num_cubes_y):
                for k in range(num_cubes_z):
                    cube_center = center +np.array([i, j, k]) * cube_size - 0.5*extent
                    cube = o3d.geometry.TriangleMesh.create_box(width=cube_size, height=cube_size, depth=cube_size)
                    cube.compute_vertex_normals()
                    cube.translate(cube_center)
                    vertices = np.asarray(cube.vertices)
                    # print(vertices)
                    # 判断这个cube内有没有点
                    range_xyz = [[vertices[0][0],vertices[-1][0]],[vertices[0][1],vertices[-1][1]],[vertices[0][2],vertices[-1][2]]]
                    points = np.asarray(point_cloud.points)
                    colors = np.asarray(point_cloud.colors)
                    index = (points[:,0] < range_xyz[0][1])&(points[:,0] > range_xyz[0][0])&(points[:,1] < range_xyz[1][1])&(points[:,1] > range_xyz[1][0])&(points[:,2] < range_xyz[2][1])&(points[:,2] > range_xyz[2][0])
                    points = points[index]

                    # points = points[(points[:,1] < range_xyz[1][1])&(points[:,1] > range_xyz[1][0])]
                    # points = points[(points[:,2] < range_xyz[2][1])&(points[:,2] > range_xyz[2][0])]

                    if len(points) == 0:
                        # continue
                        c = 0 # 空
                    else:
                        colors = colors[index]
                        color = np.mean(colors, axis=0)
                        cube.paint_uniform_color(color)

                        c_list = {1:0,2:0,3:0} # 0-空，1-地面，2-背景，3-动态/车
                        for color in colors:
                            if color[0] == 0.1:
                                c_list[1] = c_list[1]+1
                            elif color[0] == 0.2:
                                c_list[2] = c_list[2]+1
                            elif color[0] == 1:
                                c_list[3] = c_list[3]+1
                        a = max(c_list[1],c_list[2],c_list[3])
                        if a == c_list[1]:
                            c = 1
                        elif a == c_list[2]:
                            c = 2
                        elif a == c_list[3]:
                            c = 3
                    save_cube = [int(cube_center[0]),int(cube_center[1]),int(cube_center[2]),int(c)]
                    save_cubes.append(save_cube)



                    # color,all_in_points = No_points(vertices,point_cloud,all_in_points)
                    # if color is None:
                    #     continue
                    # else:
                    #     cube.paint_uniform_color(color)

                    # cube.paint_uniform_color([0.0, 1.0, 0.0])  # 设置颜色为绿色
                    cubes.append(cube)
                    lineset = draw_line(vertices)
                    cube_linesets.append(lineset)

                    if len(cubes) % 10 == 0:
                        print("len(cubes) =",len(cubes),", schedule =","%.3f"%(i/num_cubes_x),", all_schedule =","%.3f"%(num/all_num))
            #         break
            #     break
            # break

        print("len(final_cubes) =",len(cubes))

        # draw(point_cloud,cubes,cube_linesets,save_path)
        
        save_cubes = np.array(save_cubes)
        np.save(save_path, save_cubes)
        print(save_path,"is OK !")
        # break

if __name__ == "__main__":
    main()
```

## 六、导出栅格化可视化视频
要求环视相机视频与栅格化视频同时展示

生成空白幕布，将环视相机图片与栅格化图片分别排列，导出单帧图像，最后合成视频

代码为cat_images.py
```python
import cv2
import numpy as np
import os

# input = '/home/mayc/codes_tld_tsr_od/codes_tld_tsr_od/data/20220108T173053_8N3824/output/output_pro_image'
# output = '/home/mayc/codes_tld_tsr_od/codes_tld_tsr_od/data/20220108T173053_8N3824/output/all_image'
# output_video = '/home/mayc/codes_tld_tsr_od/codes_tld_tsr_od/data/20220108T173053_8N3824/output/output_video'
# camera_lists = ['FW_1.7','FN','FL_1.7','FR_1.7','RN','RL_1.7','RR_1.7','bev_images']
# all_pic_path=[]
# pic_path = []
# image_list = []
# rows = 3
# cols = 3
# width = 6528
# height = 3672
# print("imcoming")
# # sys.stdout.flush()
# for cam in camera_lists:
#     pic_path.clear()
#     cam_pic = input + '/' + cam
#     for filename in sorted(os.listdir(cam_pic)):
#         # 检查文件是否是图片
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             pic_path.append(os.path.join(cam_pic, filename))
#     all_pic_path.append(pic_path.copy())
# for n in range(590):
#     # sys.stdout.flush()
#     i = -1
#     image = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)
#     for pic_path in all_pic_path:
#         i = i + 1
#         row = i // cols
#         col = i % cols 
#         image_temp=cv2.imread(pic_path[n])
#         if i!=7:
#             image_temp=cv2.resize(image_temp,(width,height))
#             image[row * height:(row + 1) * height, col * width:(col + 1) * width, :] =image_temp
#         if i == 7:
#             image_temp=cv2.resize(image_temp,(2*width,height))
#             image[row * height:(row + 1) * height, col * width:(col + 2) * width, :] =image_temp
#     image_last = cv2.resize(image,(width,height))
#     output_path = output+"/"+f"{n:03d}"+".jpg"
#     cv2.imwrite(output_path,image_last)
#     print(n)

def get_camera(camera_dir,i,timestamp):
    FW_dir = camera_dir+'/'+ i
    names_FW = os.listdir(FW_dir)
    for name_FW in names_FW:
        if name_FW[:-8] == timestamp:
            break
    FW_path = camera_dir+'/'+ i +'/'+name_FW
    FW_image = cv2.imread(FW_path)
    FW_image = cv2.resize(FW_image,(888,500))

    return FW_image



camera_dir = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/datasets/camera"
occupany_dir = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/datasets/occupany"
save_dir = "/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/z_others/1_occupancy/datasets/occ_cat"
camera_lists = ['FW_1.7','FN','FL_1.7','FR_1.7','RL_1.7','RR_1.7','RN','RN']
# camera_position = [[0,0+500,0,0+888],[0,0+500,888+800,888*2+800],
#                    [500,500+500,0,0+888],[500,500+500,888+800,888*2+800],
#                    [500*2,500*2+500,0,0+888],[500*2,500*2+500,888+800,888*2+800],
#                    [500*3,500*3+500,0,0+888],[500*3,500*3+500,888+800,888*2+800]]
camera_position = [[0,0+500,0,0+888],[0,0+500,888+1200,888*2+1200],
                   [500,500+500,0,0+888],[500,500+500,888+1200,888*2+1200],
                   [500*2,500*2+500,0,0+888],[500*2,500*2+500,888+1200,888*2+1200],
                   [500*3,500*3+500,0,0+888],[500*3,500*3+500,888+1200,888*2+1200]]

occupany_names = os.listdir(occupany_dir)
for name in occupany_names:
    occupany_path = occupany_dir+'/'+name
    occupany_image = cv2.imread(occupany_path)
    
    save_path = save_dir+'/'+name

    image_size = [2000,1200+888*2] # [x,y]
    image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8)*255
    # occupany_image = cv2.resize(occupany_image,(800,500)) # [w,h]
    # image[700:700+500, 888:888+800] = occupany_image
    
    # occupany_image = cv2.resize(occupany_image,(1600,1000)) # [w,h]
    # image[500:500+1000, 488:488+1600] = occupany_image
    occupany_image = cv2.resize(occupany_image,(2400,1500)) # [w,h]
    image[500:500+1500, 288:288+2400] = occupany_image
    timestamp = name[:-8]

    # FW
    for i in range(len(camera_lists)):
        camera_image = get_camera(camera_dir,camera_lists[i],timestamp)
        image[camera_position[i][0]:camera_position[i][1], camera_position[i][2]:camera_position[i][3]] = camera_image
        # break
    
    cv2.imwrite(save_path,image)
    print(save_path,"is OK !")
    # cv2.imshow("Composite Image", image)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()
    # break
```
单帧可视化示意
![1641634400 939636](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/644c1f0c-f0f8-42c6-a226-7e979d3b6c4b)


