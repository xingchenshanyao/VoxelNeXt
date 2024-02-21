# DeFine
202402最新场景流方法复现

参考论文：[DeFlow: Decoder of Scene Flow Network in Autonomous Driving](https://readpaper.com/pdf-annotate/note?pdfId=2162246923133175296)

参考仓库：[DeFlow](https://github.com/KTH-RPL/DeFlow)

## 一、配置环境
复制已有环境VoxelNeXt，另存为DeFine，修改DeFine/bin/pip中
```
#!/home/xingchen/anaconda3/envs/VoxelNeXt/bin/python
```
为
```
#!/home/xingchen/anaconda3/envs/DeFine/bin/python
```
 查看environment.yaml
 ```
  - python=3.8
  - pytorch
  - torchvision
  - lightning==2.0.1
  - nvidia::cudatoolkit=11.4
  - conda-forge/label/rust_dev::rust
  - numba
  - numpy
  - pandas
  - pip
  - scipy
  - tqdm
  - h5py
  - wandb
  - omegaconf
  - hydra-core
  - open3d==0.16.0
```
 pip安装其中缺少的model
 ```
pip install lightning==2.0.1
pip install omegaconf
pip install hydra-core
pip install wandb
```

## 二、准备数据集
使用Argoverse 2 sensers数据集，由于Argoverse 2 sensers过大(1T)，选取[val Part1(52.9G)](https://s3.amazonaws.com/argoverse/datasets/av2/tars/sensor/val-000.tar)下载

