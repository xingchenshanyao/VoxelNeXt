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
pip install opencv-python timm diffdist h5py sklearn ftfy
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
