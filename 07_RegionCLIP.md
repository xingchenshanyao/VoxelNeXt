# RegionCLIP
记录复现RegionCLIP的流程及各种BUG与解决措施

参考来源

https://github.com/microsoft/RegionCLIP

## 一、安装环境
参考仓库中docs/INSTALL.md
```
# environment
conda create -n regionclip python=3.9
source activate regionclip
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# RegionCLIP
git clone git@github.com:microsoft/RegionCLIP.git
python -m pip install -e RegionCLIP # BUG1

# other dependencies
pip install opencv-python timm diffdist h5py sklearn ftfy # BUG2
pip install git+https://github.com/lvis-dataset/lvis-api.git
```
出现BUG1
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
出现BUG2
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

## 二、代码测试
### a. 配置权重
于 https://drive.google.com/drive/folders/1hzrJBvcCrahoRcqJRqzkIGFO_HUSJIii ，下载权重(1.6G)

将下载的文件夹重命名为pretrained_ckpt，放在RegionCLIP文件夹下
### b. 配置数据集
使用LVIS验证集标签，官网 https://www.lvisdataset.org/dataset 下载Validation set(192MB)

```

```
