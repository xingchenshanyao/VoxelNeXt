# Debug_VoxelNeXt_demo_test_train
本文为调试代码(细读代码)的记录，按kitti_dataset、nuscenes_dataset、demo、test、train的顺序进行

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
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict) # 检测结果

            V.draw_scenes( # 绘制检测框
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')
```

