# YOLOv8
使用YOLOv8实现nuScenes图像检测，并将3Dbbox框与图像2Dbox匹配以确定bbox类别

参考来源：

https://blog.csdn.net/weixin_44791964/article/details/129978504

https://blog.csdn.net/Tracy_Baker/article/details/121652716

https://blog.csdn.net/qq_34972053/article/details/111315493

## 调试YOLOv8
与YOLOv5，YOLOv7类似，不再赘述，参考宝藏up [Bubbliiiing](https://blog.csdn.net/weixin_44791964/article/details/129978504)

![2024-01-04 09-54-45屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/c94ed43a-40dc-4184-aa62-f1d349b9791d)

## NuScenes转格式
### a. nuScenes to json
NuScenes的标注信息为3D框格式，需要先用官方工具nuscenes-devkit将其转换为2D框格式
```
cd /home/xingchen/Study/4D_GT/VoxelNeXt_pipeline
git clone https://github.com/nutonomy/nuscenes-devkit
cd nuscenes-devkit/python-sdk/nuscenes/scripts
python3 export_2d_annotations_as_json.py --dataroot /home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/data/nuscenes/v1.0-mini --version v1.0-mini --filename 2D-box.json
```
得到2D-box.json

### b. json to txt
将json格式转成COCO数据集的txt格式

需要注意nuScenes数据集中类别共23类，仅需提取为car、person、bicycle三类

![image](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/4182da2a-88b0-425d-8963-740fa064778d)

```
git clone https://github.com/AlizadehAli/2D_label_parser
```
修改2D_label_parser/label_parser.py为
```
import os
from os import walk, getcwd
import json
from typing import List, Any
from PIL import Image
from tqdm import tqdm
import argparse

__author__ = "Ali Alizadeh"
__email__ = 'aalizade@ford.com.tr / a.alizade@live.com'
__license__ = 'AA_Parser'


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='YOLOv3 json, Berkeley Deep Drive dataset (BDD100K), nuscenes 2D labels to txt-label format for '
                    'yolov3 darknet NN model')
    parser.add_argument("-dt", "--data_type",
                        default="nuscenes",
                        help="data type of interest; yolo, bdd, nuscenes")
    parser.add_argument("-l", "--label_dir", default="2D_label_parser/labels/",
                        help="root directory of the labels for YOLO json file, Berkeley Deep Drive (BDD) json-file, "
                             "nuscenes")
    parser.add_argument("-s", "--save_dir", default="2D_label_parser/target_labels/",
                        help="path directory to save the the converted label files")
    parser.add_argument("-i", "--image_dir",
                        default=None, required=False,
                        help="path where the images are located to BDD100K, nescenes, etc.")
    parser.add_argument("-o", "--output_dir",
                        default=None, required=False,
                        help="output directory to save the manipulated image files")
    args = parser.parse_args()
    return args


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def collect_bdd_labels(bdd_label_path):
    bdd_json_list = []
    for file in tqdm(os.listdir(bdd_label_path)):
        if file.endswith(".json"):
            bdd_json_list.append(file)
    return bdd_json_list


def sync_labels_imgs(label_path, img_path):
    for path, subdirs, files in tqdm(os.walk(img_path)):
        for file in tqdm(files):
            if file.lower().endswith('jpg'):
                image_folders_path = img_path + path.split('/')[-1]
                image_path = os.path.join(image_folders_path, file)
                image_path = image_path.split('.')[0] + '.txt'
                if not os.path.isdir(image_path):
                    os.remove(image_path)


def write_training_data_path_synced_with_labels(img_path):
    with open('nuscenes_training_dataPath.txt', 'w') as train_data_path:
        for path, subdirs, files in os.walk(img_path):
            for file in files:
                if file.lower().endswith('png'):
                    full_path = os.path.join(path, file)
                    full_path = full_path.split('.')[0] + '.txt'
                    full_path = 'images/' + path.split('/')[-1] + '/' + full_path.split('/')[-1]
                    train_data_path.write(str(full_path) + os.linesep)
        train_data_path.close()


def bdd_parser(bdd_label_path):
    bdd_json_list = collect_bdd_labels(bdd_label_path)
    label_data: List[Any] = []
    for file in tqdm(bdd_json_list):
        label_data.append(json.load(open(bdd_label_path + file)))
    return label_data


def yolo_parser(json_path, targat_path):
    json_backup = "./json_backup/"

    wd = getcwd()
    list_file = open('%s_list.txt' % (wd), 'w')

    json_name_list = []
    for file in tqdm(os.listdir(json_path)):
        if file.endswith(".json"):
            json_name_list.append(file)

    """ Process """
    for json_name in tqdm(json_name_list):
        txt_name = json_name.rstrip(".json") + ".txt"
        """ Open input text files """
        txt_path = json_path + json_name
        print("Input:" + txt_path)
        txt_file = open(txt_path, "r")

        """ Open output text files """
        txt_outpath = targat_path + txt_name
        print("Output:" + txt_outpath)
        txt_outfile = open(txt_outpath, "a")

        """ Convert the data to YOLO format """
        lines = txt_file.read().split('\r\n')
        for idx, line in tqdm(enumerate(lines)):
            if ("lineColor" in line):
                break
        if ("label" in line):
            x1 = float(lines[idx + 5].rstrip(','))
            y1 = float(lines[idx + 6])
            x2 = float(lines[idx + 9].rstrip(','))
            y2 = float(lines[idx + 10])
            cls = line[16:17]

            """ in case when labelling, points are not in the right order """
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
    ymax = max(y1, y2)
    img_path = str('%s/dataset/%s.jpg' % (wd, os.path.splitext(json_name)[0]))

    im = Image.open(img_path)
    w = int(im.size[0])
    h = int(im.size[1])

    print(w, h)
    print(xmin, xmax, ymin, ymax)
    b = (xmin, xmax, ymin, ymax)
    bb = convert((w, h), b)
    print(bb)
    txt_outfile.write(cls + " " + " ".join([str(a) for a in bb]) + '\n')

    os.rename(txt_path, json_backup + json_name)  # move json file to backup folder

    """ Save those images with bb into list"""
    if txt_file.read().count("label") != 0:
        list_file.write('%s/dataset/%s.jpg\n' % (wd, os.path.splitext(txt_name)[0]))

    list_file.close()


def nuscenes_parser(label_path, target_path, img_path):
    json_backup = "json_backup/"
    wd = getcwd()

    dict = {
        'human.pedestrian.adult': '1',
        'human.pedestrian.child': '1',
        'human.pedestrian.wheelchair': '1',
        'human.pedestrian.stroller': '1',
        'human.pedestrian.personal_mobility': '1',
        'human.pedestrian.police_officer': '1',
        'human.pedestrian.construction_worker': '1',
        'vehicle.bicycle': '2',
        'vehicle.motorcycle': '2',
        'vehicle.car': '0',
        'vehicle.bus.bendy': '0',
        'vehicle.bus.rigid': '0',
        'vehicle.truck': '0',
        'vehicle.emergency.ambulance': '0',
        'vehicle.emergency.police': '0',
        'vehicle.construction': '0',  # 工程用车，挖掘机啥的
        'vehicle.trailer': '0'
        # 'animal': '9',
        # 'movable_object.barrier': '10',
        # 'movable_object.trafficcone': '10',
        # 'movable_object.pushable_pullable': '10',
        # 'movable_object.debris': '10',
        # 'tatic_object.bicycle_rack': '11'
    }

    json_name_list = []
    for file in tqdm(os.listdir(label_path)):
        if file.endswith(".json"):
            json_name_list.append(file)
            data = json.load(open(label_path + file))

            # Aggregate the bounding boxes associate with each image
            unique_img_names = []
            for i in tqdm(range(len(data))):
                unique_img_names.append(data[i]['filename'])

            unique_img_names = list(dict.fromkeys(unique_img_names))

            i: int
            for i in tqdm(range(len(unique_img_names))):
                f = open(target_path + unique_img_names[i].split('/')[1] + '/' +
                         unique_img_names[i].split('/')[-1].split('.')[0] + '.txt', "w+")
                for idx, name in enumerate(data):
                    if unique_img_names[i] == name['filename']:
                    # 加上一个判定条件，符合上面类别才会将坐标写入txt中
                        if name['category_name'] in dict:
                            obj_class = dict[name['category_name']]
                            x, y, w, h = convert((1600, 900), name['bbox_corners'])
                            temp = [str(obj_class), str(x), str(y), str(w), str(h), '\n']
                            L = " "
                            L = L.join(temp)
                            f.writelines(L)
                f.close()
            sync_labels_imgs(target_path, img_path)
            n = open('nuscenes.names', "w+")
            n.write('pedestrian \n')
            n.write('bicycle \n')
            n.write('motorcycle \n')
            n.write('car \n')
            n.write('bus \n')
            n.write('truck \n')
            n.write('emergency \n')
            n.write('animal \n')
            n.close()
        write_training_data_path_synced_with_labels(img_path)


if __name__ == '__main__':
    args = parse_arguments()
    if args.data_type == 'yolo':
        yolo_parser(args.label_dir, args.save_dir)
    elif args.data_type == 'bdd':
        data = bdd_parser(args.label_dir)
    elif args.data_type == 'nuscenes':
        nuscenes_parser(args.label_dir, args.save_dir, args.image_dir)
    else:
        print(40 * '-')
        print('{} data is not included in this parser!'.format(args.data_type))
```
在2D_label_parser/target_labels下新建文件夹CAM_BACK、CAM_BACK_LEFT、CAM_BACK_RIGHT、CAM_FRONT、CAM_FRONT_LEFT、CAM_FRONT_RIGHT
