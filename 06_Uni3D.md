# Uni3D
记录从Uni3D论文及对应代码仓库中学习筛选的对多激光雷达数据集3D目标检测泛化课题可能有用的信息，以及本地复现Uni3D的过程

参考来源：

论文：[Uni3D: A Unified Baseline for Multi-dataset 3D Object Detection](https://readpaper.com/pdf-annotate/note?pdfId=4750598536836431873&noteId=2098620142363720192)

代码：https://github.com/PJLab-ADG/3DTrans

## 一、论文解读
### 1.1. 多数据集3D目标检测的主要挑战
#### a. 数据分布级别的差异
不同于由[0,255]的像素分布组成的2D图像数据，来自于不同数据集的3D点云数据通常是由不同传感器厂商采用不一致的点云感知范围采样获得的。

 实际上，在3D场景中难以进行多数据集3D目标检测。其原因是：传感器不一致的点云感知范围导致对于相同物体的3D检测网络感受野是不一致的，这样会阻碍网络在不同数据集中对一种类别物体的学习过程。因此，对点云进行前处理，包括不同传感器数据的点云范围对齐，传感器高度数据对齐，是进行多数据集3D目标检测的必要前处理过程。

此外，数据分布级别差异也包括不同国家、地理位置，车辆大小的差异。这种差异主要会影响到3D检测器的候选框object-size的设定，对不同场景下物体定位的偏差等。
#### b. 语义分布级别的差异
目前，不同数据集或者厂商还没有采用统一的感知类别指定标注。例如Waymo对“Vehicle”的定义是所有在地面上进行行驶的四轮车，包括卡车、货车、小汽车等。而nuScenes会采用更加细粒度的类别定义。因此要进行多数据集3D目标检测任务，还需要考虑语义相似性强但是定义不一致的类别空间。
### 1.2. Uni3D模型
![image](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/6313dde6-7250-4758-90dd-8784f37c4a55)

包括：point range alignment——点云范围对齐/坐标矫正模块，parameter-shared 3D and 2D backbones with data-level correction operation——共享参数的3D/2D骨干网络，semantic-level feature coupling-and-recoupling module——语义级特征耦合和再耦合模块，以及dataset-specific detection heads——特定数据集的检测头部，C.A.——坐标原点对齐(以减少点范围对齐的不利影响)，S.A.——统计级对齐。

Uni3D可以将不一致的点云对齐到相同点云范围下进行训练，与此同时Uni3D采用共享参数的骨干网络和特定数据集的检测头部来实现联合多数据集检测任务，这种方式可以使用一个模型同时预测多个带有不一致类别定义和语义差异的数据集，这也进一步实现了MDF(MDF多域融合是指3D目标检测器在多个数据集上进行联合训练，并且需要尽可能同时在多个数据集实现较高的检测精度)的任务要求。除此之外，Uni3D还包括两个新增模块：1）数据级校正操作和 2）语义级特征耦合-重耦合模块。
  
## 二、参考信息
### 2.1. 数据集差异
常见点云数据集检测对象尺寸差异：[Statistical Results of Object Size](https://github.com/PJLab-ADG/3DTrans/blob/master/docs/STATISTICAL_RESULTS.md)

**KITTI nuScenes Waymo for Car**
|                                             |  KITTI@Car | nuScenes@car |   Waymo@Vehicle |
|--------------------------|-------------------------------------------:|:--------------------------------------:|:-------------------------------------:|
| Number of Frames | 7481 | 28130 | 158081 |
| Number of Instances | 28742 | 339949 | 4778641 |
| #Instances per Frame | 3.80 | 12.08 | 30.22 |
| Z-axis (mean, std, min, max, median) | -0.75, 0.33, -2.71, 3.1, -0.8 | -0.93, 1.48, -9.99, 9.37, -1.02 | 0.86, 1.02, -12.64, 13.14, 0.81 |
| L (mean, std, min, max, median) | 3.88, 0.43, 2.19, 6.67, 3.88 |  4.63, 0.47, 2.4, 11.52, 4.59 | 4.8, 1.16, 0.07, 19.69, 4.63 |
| W (mean, std, min, max, median) | 1.63. 0.1, 1.14, 2.04, 1.63 |  1.96, 0.19, 1.21, 3.88, 1.94 | 2.1, 0.3, 0.32, 8.29, 2.08 |
| H (mean, std, min, max, median) | 1.53, 0.14, 1.14, 2.48, 1.50 |  1.74, 0.25, 0.93, 4.49, 1.7 | 1.78, 0.41, 0.22, 10.22, 1.69 |
| SHIFT of Z-coordinate | 1.6 = 0.81 -（-0.8） |  1.8=0.81 -（-1.02）  | \ |

**KITTI nuScenes Waymo for Pedestrian**
|                                             |  KITTI@Pedestrian | nuScenes@Pedestrian |   Waymo@Pedestrian |
|--------------------------|-------------------------------------------:|:--------------------------------------:|:-------------------------------------:|
| Number of Frames | 7481 | 28130 | 158081 |
| Number of Instances | 4487 | 161928 | 2229448 |
| #Instances per Frame | 0.60 | 5.756 | 14.10 |
| Z-axis (mean, std, min, max, median) | -0.580, 0.254, -2.058, 0.729, -0.633 | -0.688, 1.215, -8.465, 8.887, -0.737 | 0.887, 0.7 46, -7.185, 7.988, 0.873 |
| L (mean, std, min, max, median) | 0.842, 0.235, 0.200, 1.440, 0.880 |  0.729, 0.190, 0.214, 2.130, 0.708  | 0.910, 0.197, 0.089, 4.313, 0.913 |
| W (mean, std, min, max, median) | 0.660, 0.142, 0.300, 1.200, 0.650 |  0.669, 0.139, 0.221, 1.971, 0.658 | 0.856, 0.154, 0.185, 2.680, 0.846 |
| H (mean, std, min, max, median) | 1.760, 0.113, 1.140, 2.010, 1.770 |  1.770, 0.188, 0.293, 2.930, 1.784 | 1.737, 0.209, 0.310, 3.420, 1.760 |
| SHIFT of Z-coordinate | 1.506 = 0.873 - (-0.633) |  1.61 = 0.873 - (-0.737)  | \ |

**KITTI nuScenes Waymo for Cyclist**
|                                             |  KITTI@Cyclist | nuScenes@Bicycle |   Waymo@Cyclist |
|--------------------------|-------------------------------------------:|:--------------------------------------:|:-------------------------------------:|
| Number of Frames | 7481 | 28130 | 158081 |
| Number of Instances | 1627 | 8185 | 53344 |
| #Instances per Frame | 0.217 | 0.291 | 0.337 |
| Z-axis (mean, std, min, max, median) | -0.620, 0.302, -2.463, 0.552, -0.621 | -1.027, 0.975, -6.077, 4.821, -1.070 | 0.739, 0.652, -3.097, 5. 106 , 0.794 |
| L (mean, std, min, max, median) | 1.763, 0.176, 1.190, 2.170, 1.760 |  1.701, 0.252, 0.454, 2.854, 1.696 | 1.810, 0.277, 0.530, 3.524, 1.846 |
| W (mean, std, min, max, median) | 0.597, 0.124, 0.340, 0.930, 0.600 |  0.608, 0.163, 0.209, 1.661, 0.602 | 0.844, 0.115, 0.434, 1.633, 0.834 |
| H (mean, std, min, max, median) | 1.737, 0.094, 1.410, 2.090, 1.750 |  1.303, 0.352, 0.501, 2.223, 1.156 | 1.770, 0.186, 0.890, 2.470, 1.800 |
| SHIFT of Z-coordinate | 1.415 = 0.794 - (-0.621)） |  1.864 = 0.794 - (-1.070)  | \ |

## 三、代码复现
参考https://github.com/PJLab-ADG/3DTrans

克隆库后，运行
```
python setup.py develop
```
 出现bug
 ```
running develop
/home/xingchen/anaconda3/envs/3DTrans/lib/python3.8/site-packages/setuptools/command/easy_install.py:144: EasyInstallDeprecationWarning: easy_install command is deprecated. Use build and pip and other standards-based tools.
  warnings.warn(
/home/xingchen/anaconda3/envs/3DTrans/lib/python3.8/site-packages/setuptools/command/install.py:34: SetuptoolsDeprecationWarning: setup.py install is deprecated. Use build and pip and other standards-based tools.
  warnings.warn(
running egg_info
writing pcdet.egg-info/PKG-INFO
writing dependency_links to pcdet.egg-info/dependency_links.txt
writing requirements to pcdet.egg-info/requires.txt
writing top-level names to pcdet.egg-info/top_level.txt
reading manifest file 'pcdet.egg-info/SOURCES.txt'
adding license file 'LICENSE'
writing manifest file 'pcdet.egg-info/SOURCES.txt'
running build_ext
/home/xingchen/anaconda3/envs/3DTrans/lib/python3.8/site-packages/torch/utils/cpp_extension.py:398: UserWarning: There are no g++ version bounds defined for CUDA version 11.8
  warnings.warn(f'There are no {compiler_name} version bounds defined for CUDA version {cuda_str_version}')
building 'pcdet.ops.iou3d_nms.iou3d_nms_cuda' extension
Emitting ninja build file /home/xingchen/Study/4D_GT/3DTrans/build/temp.linux-x86_64-cpython-38/build.ninja...
Compiling objects...
```
推测原因是g++与gcc安装不正确

***

暂时搁置
