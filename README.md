# VoxelNeXt
本仓库仅用于记录本人复现VoxelNeXt的流程及各种BUG与解决措施
参考来源
```
https://github.com/dvlab-research/VoxelNeXt
```
```
https://blog.csdn.net/AaaA00000001/article/details/127021967
```
```
https://blog.csdn.net/weixin_52288941/article/details/133518555
```
### 部分说明
ubuntu18.04、cuda11.8、python3.8、GPU3070、GPU Driver 520.61.05、torch2.0.0+cu118
## 一、克隆VoxelNeXt仓库并进入文件夹
```
https://github.com/dvlab-research/VoxelNeXt && cd VoxelNeXt
```
## 二、安装OpenPCDet环境
### a、复制现有torch环境作为基础
例如，将anaconda3/envs/torch20230517(一个能跑通YOLOP的torch环境)文件夹另存副本，改名为VoxelNeXt
然后更改anaconda3/envs/VoxelNeXt/bin/pip中的
```
#!/home/xingchen/anaconda3/envs/torch20230517/bin/python
```
为
```
#!/home/xingchen/anaconda3/envs/VoxelNeXt/bin/python
```
### b、安装pcdet v0.5
```
git clone https://github.com/open-mmlab/OpenPCDet.git
```
### c、安装spconv-cu118
```
pip install spconv-cu118
```
其他cuda版本参照
```
https://github.com/traveller59/spconv
```
### c、运行setup文件完成
```
python setup.py develop
```
#### DUG1
运行步骤c报错
```
The detected CUDA version (9.1) mismatches the version that was used to compile PyTorch (11.8). Please make sure to use the same CUDA versions.
```
