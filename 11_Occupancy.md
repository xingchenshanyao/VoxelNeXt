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


## 三、重新可视化需求
- 将前后10帧的点云中，动态点去除，静态点在速度补偿后叠加
- 叠加后的点云数据与一组动态点叠加
- 进行栅格化

