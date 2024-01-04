# YOLOv8
使用YOLOv8实现nuScenes图像检测，并将3Dbbox框与图像2Dbox匹配以确定bbox类别

参考来源：

https://blog.csdn.net/weixin_44791964/article/details/129978504

https://blog.csdn.net/Tracy_Baker/article/details/121652716

https://blog.csdn.net/qq_34972053/article/details/111315493

## 调试YOLOv8
与YOLOv5，YOLOv7类似，不再赘述，参考宝藏up [Bubbliiiing](https://blog.csdn.net/weixin_44791964/article/details/129978504)

![2024-01-04 09-54-45屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/c94ed43a-40dc-4184-aa62-f1d349b9791d)

## NuScenes转格式
### a. nuScenes to json
NuScenes的标注信息为3D框格式，需要先用官方工具nuscenes-devkit将其转换为2D框格式
```
cd /home/xingchen/Study/4D_GT/VoxelNeXt_pipeline
git clone https://github.com/nutonomy/nuscenes-devkit
cd nuscenes-devkit/python-sdk/nuscenes/scripts
python3 export_2d_annotations_as_json.py --dataroot /home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/data/nuscenes/v1.0-mini --version v1.0-mini --filename 2D-box.json
```
得到2D-box.json

### b. json to txt
将json格式转成COCO数据集的txt格式

需要注意

