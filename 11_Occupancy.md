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

