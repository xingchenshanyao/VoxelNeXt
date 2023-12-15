# VoxelNeXt on the Server
记录VoxelNeXt在服务器上的复现过程
### 部分说明
ubuntu20.04、cuda11.6/11.3、A100*4、GPU Driver510
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
## 二、配置服务器环境
### a、
由于服务器cuda版本为11.6/11.3，无法直接上传本地cuda11.8的环境

此时要么在服务器上装cuda11.8，但是由于显卡驱动是510，最高仅支持cuda11.6，所以装cuda11.8前必须升级驱动

或者重新配置cuda11.6的环境，选择后者

在anaconda/envs中拷贝他人环境FN**
```
cd /home/**/anaconda/envs
cp -r FN** VoxelNeXt
```
修改VoxelNeXt/bin/pip，用vim指令打卡编辑
```
cd VoxelNeXt/bin
vim pip
```
打开pip配置文件后，按i进入编辑模式，将第一行改为
```
#!/home/**/anaconda3/envs/VoxelNeXt/bin/python
```
按esc退出编辑模式，按 :x 保存退出pip文件

激活VoxelNeXt并使用pip list查看当前配置
```
conda activate VoxelNeXt
pip list
```
得知
```
torch                     1.12.1+cu113
```
此时的环境是一个支持其他模型的cuda11.3的pytorch1.12.1环境，依赖库肯定不全，需要补充
#### BUG 1
使用pip list输出pip列表后，报错
```
WARNING: The repository located at ** is not a trusted or secure host and is being ignored. If this repository is available via HTTPS we recommend you use HTTPS instead, otherwise you may silence this warning and allow it anyway with '--trusted-host **'.
```
搜索后得知，之后使用pip安装依赖需要添加'--trusted-host **'，这个BUG暂时没有影响
```
pip install name --trusted-host **
```
