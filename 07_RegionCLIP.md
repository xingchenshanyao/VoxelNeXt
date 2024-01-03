# RegionCLIP
记录本地和服务器复现RegionCLIP的流程及各种BUG与解决措施

参考来源

https://github.com/microsoft/RegionCLIP

https://blog.csdn.net/wzk4869/article/details/131731416

## 一、安装环境
参考仓库中docs/INSTALL.md
```
# environment
conda create -n RegionCLIP python=3.9
conda activate RegionCLIP
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # BUG3 !!!

# RegionCLIP
git clone git@github.com:microsoft/RegionCLIP.git
python -m pip install -e RegionCLIP # BUG1

# other dependencies
pip install opencv-python timm diffdist h5py sklearn ftfy # BUG2
pip install git+https://github.com/lvis-dataset/lvis-api.git
```
### BUG1
```
Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [60 lines of output]
      No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda-11.8'
      /home/xingchen/anaconda3/envs/RegionCLIP/lib/python3.9/site-packages/setuptools/dist.py:315: SetuptoolsDeprecationWarning: Invalid version: 'RegionCLIP'.
……
```
解决方式
```
pip install --upgrade pip setuptools==57.5.0
```
### BUG2
```
Collecting sklearn
  Using cached https://mirrors.aliyun.com/pypi/packages/b9/0e/b2a4cfaa9e12b9ca4c71507bc26d2c99d75de172c0088c9835a98cf146ff/sklearn-0.0.post10.tar.gz (3.6 kB)
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error

  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [18 lines of output]
      The 'sklearn' PyPI package is deprecated, use 'scikit-learn'
      rather than 'sklearn' for pip commands.
……
```
sklearn已被弃用，应下载scikit-learn
```
pip install opencv-python timm diffdist h5py scikit-learn ftfy 
```
安装时出现警告，暂时忽略
```
DEPRECATION: detectron2 RegionCLIP has a non-standard version number. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of detectron2 or contact the author to suggest that they release a version with a conforming version number. Discussion can be found at https://github.com/pypa/pip/issues/12063
```
### BUG3
采用该指令安装pytorch，发现torch.cuda.is_available() = False，所以建议直接改成官网指令安装pytorch

在Pytorch官网查看cuda11.8对应的PyTorch下载指令
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
补充安装cudatoolkit=11.3
```
conda install cudatoolkit=11.3
```
## 二、代码测试
### a. 配置权重
于 https://drive.google.com/drive/folders/1hzrJBvcCrahoRcqJRqzkIGFO_HUSJIii ，下载权重(1.6G)

将下载的文件夹解压后，重命名为pretrained_ckpt，放在RegionCLIP文件夹下
### b. 配置数据集
使用LVIS验证集标签，官网 https://www.lvisdataset.org/dataset 下载Validation set的标签(192MB)

在RegionCLIP/datasets下新建lvis，并把下载好的文件解压lvis_v1_val.json放在此处

### c. 代码执行
```
python ./tools/train_net.py --eval-only --num-gpus 1 --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb_rn50x4.pth MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml MODEL.CLIP.TEXT_EMB_DIM 640 MODEL.RESNETS.DEPTH 200 MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 
```
```
python ./tools/train_net.py \
--eval-only \
--num-gpus 1 \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb_rn50x4.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
```
### BUG4
```
module 'PIL.Image' has no attribute 'LINEAR'
```
解决方法，降低Pillow版本
```
pip install Pillow==8.4.0 -i https://pypi.mirrors.ustc.edu.cn/simple/
```
### BUG5
本地显存不够，只能放服务器上解决了
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 802.00 MiB. GPU 0 has a total capacty of 7.77 GiB of which 484.94 MiB is free. Including non-PyTorch memory, this process has 6.52 GiB memory in use. Of the allocated memory 4.37 GiB is allocated by PyTorch, and 850.97 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```
## 三、服务器环境配置和代码测试
上传代码到服务器/home/test/users/xuzeyuan中

.38服务器为cuda11.6，有torch-1.9.1+cu111-cp38-cp38-linux_x86_64.whl和torchvision-0.10.1+cu111-cp38-cp38-linux_x86_64.whl，配置环境
```
conda create -n RegionCLIP python=3.8
conda activate RegionCLIP
pip install torch-1.9.1+cu111-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.10.1+cu111-cp38-cp38-linux_x86_64.whl
pip install torchaudio==0.9.1 # 与torch-1.9.1+cu111匹配
conda install cudatoolkit=10.0 # 随便调了一个老版本的

cd /home/test/users/xuzeyuan
pip install --upgrade pip setuptools==57.5.0
python -m pip install -e RegionCLIP
pip install opencv-python timm diffdist h5py scikit-learn ftfy

pip install git+https://github.com/lvis-dataset/lvis-api.git
```
下载https://github.com/lvis-dataset/lvis-api.git
```
git clone https://github.com/lvis-dataset/lvis-api.git
```
上传lvis-api到服务器/home/test/users/xuzeyuan中
```
pip install lvis-api/
```
测试代码
```
python ./tools/train_net.py --eval-only --num-gpus 1 --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb_rn50x4.pth MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml MODEL.CLIP.TEXT_EMB_DIM 640 MODEL.RESNETS.DEPTH 200 MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18
```
### BUG6
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
解决措施
```
pip install opencv-python-headless
```
### BUG7
```
module 'PIL.Image' has no attribute 'LINEAR'
```
解决方法，降低Pillow版本
```
pip install Pillow==8.4.0 -i https://pypi.mirrors.ustc.edu.cn/simple/
```
运行通过，成功在output/inference下生成lvis_instances_results.json
![2024-01-03 12-53-49屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/94d46e1b-20b4-415e-9c1f-52d487d9d413)
## 四、本地可视化
根据生成的文件output/inference/lvis_instances_results.json进行可视化
```
python ./tools/visualize_json_results.py --input ./output/inference/lvis_instances_results.json --output ./output/regions --dataset lvis_v1_val_custom_img --conf-threshold 0.05 --show-unique-boxes --max-boxes 25 --small-region-px 8100\
```
```
python ./tools/visualize_json_results.py \
--input ./output/inference/lvis_instances_results.json \
--output ./output/regions \
--dataset lvis_v1_val_custom_img \
--conf-threshold 0.05 \
--show-unique-boxes \
--max-boxes 25 \
--small-region-px 8100\ 
```
可视化效果

![sample_img1](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/47d84170-906b-4c01-a939-8a02d716e4ac)
![sample_img2](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/1d0816ba-bd72-4a9c-a443-a59681e8e765)

## 五、测试nuScenes图片
拿官方提供的权重与nuScenes数据集中的图片直接进行检测，效果不佳，可视化结果为

![n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385097362404](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/6bfd9a18-e867-4687-a909-1c4533e36c2c)
![n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385093612404](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/beb6da71-2f40-4496-8913-ba0b5447c3d1)


