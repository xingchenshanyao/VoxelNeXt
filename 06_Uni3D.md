# Uni3D Learning
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


  
