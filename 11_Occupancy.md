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
ctr.set_lookat(np.array([55.0, 55, 55.0]))
ctr.set_up((1, -1, 1))  # 指向屏幕上方的向量
ctr.set_front((-1, 1, 1))  # 垂直指向屏幕外的向量
# ctr.set_zoom(0.001) # 控制远近
vis.update_geometry(pcd)
vis.poll_events()
vis.update_renderer()
```
经反复查询后，得知open3d==0.17.0存在BUG(kusi)，需要退回0.16.0
```
pip install open3d==0.16.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 三、重新可视化需求
- 将前后10帧的点云中，动态点去除，静态点在速度补偿后叠加
- 叠加后的点云数据与一组动态点叠加
- 进行栅格化

