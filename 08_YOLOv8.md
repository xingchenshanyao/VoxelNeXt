# YOLOv8
使用YOLOv8实现nuScenes图像检测，并将3Dbbox框与图像2Dbox匹配以确定bbox类别

参考来源：

https://blog.csdn.net/weixin_44791964/article/details/129978504

https://blog.csdn.net/Tracy_Baker/article/details/121652716

## 调试YOLOv8
与YOLOv5，YOLOv7类似，不再赘述，参考宝藏up [Bubbliiiing](https://blog.csdn.net/weixin_44791964/article/details/129978504)

![2024-01-04 09-54-45屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/c94ed43a-40dc-4184-aa62-f1d349b9791d)

## NuScenes转格式
NuScenes的标注信息为3D框格式，需要先用官方工具nuscenes-devkit将其转换为2D框格式

