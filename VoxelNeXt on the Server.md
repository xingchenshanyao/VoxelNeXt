# VoxelNeXt on the Server
记录VoxelNeXt在服务器上的复现过程
### 部分说明
ubuntu20.04、cuda11.6、A100*4
## 一、上传VoxelNeXt程序到服务器内
### a、安装Filezilla
由于服务器无法连接外网(无法直接git clone)，必须通过Filezilla上传程序
```
sudo apt-get install filezilla
```
### b、将VoxelNeXt文件夹传输到服务器指定位置
```
Path = /home/**/**yuan
cd Path
```
