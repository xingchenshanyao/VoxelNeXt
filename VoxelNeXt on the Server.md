# VoxelNeXt on the Server
记录VoxelNeXt在服务器上的复现过程
参考来源

https://github.com/dvlab-research/VoxelNeXt

https://blog.csdn.net/AaaA00000001/article/details/127021967

https://blog.csdn.net/weixin_52288941/article/details/133518555

https://blog.csdn.net/weixin_45811857/article/details/124457280

### 部分说明
ubuntu20.04、cuda11.6、A100*4
## 一、上传VoxelNeXt程序到服务器内
### a、安装Filezilla
由于服务器无法连接外网(无法直接git clone)，必须通过Filezilla上传程序
```
sudo apt-get install filezilla
```
