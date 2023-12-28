# Debug_VoxelNeXt_demo_test_train
本文为调试代码(细读代码)的记录，按demo、(test、train)的顺序进行

调试demo、test、train时，配置文件直接调用cbgs_voxel0075_voxelnext.yaml与nuscenes_dataset.yaml

为便于代码分析，调试过程在本地完成

参考来源：

https://blog.csdn.net/xuchaoxin1375/article/details/117402006
### 部分说明
ubuntu18.04、cuda11.8、python3.8、GPU3070、GPU Driver 520.61.05、torch2.0.0+cu118
***
## 一、demo
原运行指令
```
cd tools
python demo2.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ckpt /home/xingchen/Study/4D_GT/VoxelNeXt/output/nuscenes_models_All/cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_20.pth --data_path /home/xingchen/Study/4D_GT/VoxelNeXt/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025.pcd.bin
```
为直接debug demo.py，需要修改tools/demo2.py(注意为nuScenes的可视化在demo2.py中)第70行中
```
parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
```
  为
```
parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml',
                        help='specify the config for demo')
parser.add_argument('--data_path', type=str, default='/home/xingchen/Study/4D_GT/VoxelNeXt/data/nuscenes/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025.pcd.bin',
                        help='specify the point cloud data file or directory')
parser.add_argument('--ckpt', type=str, default='/home/xingchen/Study/4D_GT/VoxelNeXt/output/nuscenes_models_All/cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_20.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
```
随后发现cd tools文件夹后，点击调试debug，vscode会自动cd回父目录，导致运行错误，解决方法如下

![2023-12-21 10-33-45屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/2b0f11b3-8b88-4aa8-a967-8017abac327e)
### 1.1. main()
```python
def main():
    args, cfg = parse_config() # args为输入路径，cfg为配置文件
    logger = common_utils.create_logger() # 创建日志
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset) # 加载模型
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True) # 加载参数
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset): # data_dict = {'points':[[x,y,z,intensity],...],'frame_id':0,'use_lead_xyz':True,'voxels':[[[x,y,z,intensity]*10],...],'voxel_coords':[[z,x,y]?,...],'voxel_num_points':[1,...]}
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict]) #  data_dict = {'points': [[0,x,y,z,intensity],...], 'frame_id':[0], 'use_lead_xyz':[True], 'voxels':[[x,y,z,intensity],...], 'voxel_coords':[[batch_size = 0,z,x,y]?,...], 'voxel_num_points':[1,...], 'batch_size': 1}
            load_data_to_gpu(data_dict) # 把data_dict放到gpu上,array变成tensor的格式
            pred_dicts, _ = model.forward(data_dict) # 检测结果 
            # pred_dicts = [{'pred_boxes':[9个参数],'pred_scores':[0.2692,...],'pred_labels':[1,...],'pred_ious':[None,...]}]
            # eg. [{'pred_boxes':tensor([[ 1.2003e+01,  3.4673e+01, -1.9518e-01,  4.5309e+00,  1.9496e+00,1.6281e+00, -1.2367e-01, -9.0418e-05,  4.2779e-05]], device='cuda:0'),'pred_scores':[0.2692],'pred_labels':[1],'pred_ious':[None, None, None, None, None, None]}]
            
            
            # # 为确定pred_boxes的9个参数分别是啥：
            # # 激光雷达坐标为原点 x(red) y(green) z(blue) 长(x方向) 宽(y方向) 高(z方向) 弧度制表示的与+x,+y,+z夹角(但是后续+y+z夹角被置0了)
            # pred_box0 = [ 0,  0, 0, 10,  5,  2, 0, 0, 1]
            # pred_box1 = [ -3.19,  -22.85, -1.89, 4.27,  1.83,  1.66, -1.42, 0, 0]
            # pred_box2 = [ -3.19,  -22.85, -1.89, 4.27,  1.83,  1.66, -1.42, 1.28, -8.38]
            # Bebug: pred_boxes中的最后一个框不会被画出来
            # pred_boxes = [pred_box0,pred_box1,pred_box2]
            # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # pred_boxes = torch.tensor(pred_boxes).to(device)
            # # pred_dicts = [{'pred_boxes':pred_boxes,'pred_scores':torch.tensor([0.1,0.1,0.2,0.3,0.4]).to(device),'pred_labels':torch.tensor([1,1,1,1,1]).to(device),'pred_ious':[None, None, None, None, None, None]}]
            # pred_dicts = [{'pred_boxes':pred_boxes,'pred_scores':torch.tensor([0.4,0.5,0.5]).to(device),'pred_labels':torch.tensor([1,1,1]).to(device),'pred_ious':[None, None, None, None, None, None]}]

            V.draw_scenes( # 绘制检测框
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')
```

