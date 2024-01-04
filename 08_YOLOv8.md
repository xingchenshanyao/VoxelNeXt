# YOLOv8
使用YOLOv8实现nuScenes图像检测，并将3Dbbox框与图像2Dbox匹配以确定bbox类别

参考来源：

https://blog.csdn.net/weixin_44791964/article/details/129978504

https://blog.csdn.net/Tracy_Baker/article/details/121652716

https://blog.csdn.net/qq_34972053/article/details/111315493

## 一、调试YOLOv8
与YOLOv5，YOLOv7类似，不再赘述，参考宝藏up [Bubbliiiing](https://blog.csdn.net/weixin_44791964/article/details/129978504)

![2024-01-04 09-54-45屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/c94ed43a-40dc-4184-aa62-f1d349b9791d)

## 二、NuScenes转格式
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
```
python label_parse.py
```
得到txt格式的标签

### c. txt to xml
从COCO的txt格式到VOC的xml格式

将所有图片和txt标签分别放在同一个文件夹下，修改路径，运行以下脚本即可
```
#############################################################
# #### txt to xml
#############################################################
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
from PIL import Image
import cv2


dict = {
    '0': 'car',
    '1': 'person',
    '2': 'bicycle'
}


class Xml_make(object):
    def __init__(self):
        super().__init__()

    def __indent(self, elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.__indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _imageinfo(self, list_top):
        annotation_root = ET.Element('annotation')
        annotation_root.set('verified', 'no')
        tree = ET.ElementTree(annotation_root)
        '''
        0:xml_savepath 1:folder,2:filename,3:path
        4:checked,5:width,6:height,7:depth
        '''
        folder_element = ET.Element('folder')
        folder_element.text = list_top[1]
        annotation_root.append(folder_element)

        filename_element = ET.Element('filename')
        filename_element.text = list_top[2]
        annotation_root.append(filename_element)

        path_element = ET.Element('path')
        path_element.text = list_top[3]
        annotation_root.append(path_element)

        checked_element = ET.Element('checked')
        checked_element.text = list_top[4]
        annotation_root.append(checked_element)

        source_element = ET.Element('source')
        database_element = SubElement(source_element, 'database')
        database_element.text = 'Unknown'
        annotation_root.append(source_element)

        size_element = ET.Element('size')
        width_element = SubElement(size_element, 'width')
        width_element.text = str(list_top[5])
        height_element = SubElement(size_element, 'height')
        height_element.text = str(list_top[6])
        depth_element = SubElement(size_element, 'depth')
        depth_element.text = str(list_top[7])
        annotation_root.append(size_element)

        segmented_person_element = ET.Element('segmented')
        segmented_person_element.text = '0'
        annotation_root.append(segmented_person_element)

        return tree, annotation_root

    def _bndbox(self, annotation_root, list_bndbox):
        for i in range(0, len(list_bndbox), 8):
            object_element = ET.Element('object')
            name_element = SubElement(object_element, 'name')
            name_element.text = list_bndbox[i]

            # flag_element = SubElement(object_element, 'flag')
            # flag_element.text = list_bndbox[i + 1]

            pose_element = SubElement(object_element, 'pose')
            pose_element.text = list_bndbox[i + 1]

            truncated_element = SubElement(object_element, 'truncated')
            truncated_element.text = list_bndbox[i + 2]

            difficult_element = SubElement(object_element, 'difficult')
            difficult_element.text = list_bndbox[i + 3]

            bndbox_element = SubElement(object_element, 'bndbox')
            xmin_element = SubElement(bndbox_element, 'xmin')
            xmin_element.text = str(list_bndbox[i + 4])

            ymin_element = SubElement(bndbox_element, 'ymin')
            ymin_element.text = str(list_bndbox[i + 5])

            xmax_element = SubElement(bndbox_element, 'xmax')
            xmax_element.text = str(list_bndbox[i + 6])

            ymax_element = SubElement(bndbox_element, 'ymax')
            ymax_element.text = str(list_bndbox[i + 7])

            annotation_root.append(object_element)

        return annotation_root

    def txt_to_xml(self, list_top, list_bndbox):
        tree, annotation_root = self._imageinfo(list_top)
        annotation_root = self._bndbox(annotation_root, list_bndbox)
        self.__indent(annotation_root)
        tree.write(list_top[0], encoding='utf-8', xml_declaration=True)


def txt_2_xml(source_path, xml_save_dir, txt_dir):
    COUNT = 0
    for folder_path_tuple, folder_name_list, file_name_list in os.walk(source_path):
        for file_name in file_name_list:
            file_suffix = os.path.splitext(file_name)[-1]
            if file_suffix != '.jpg':
                continue
            list_top = []
            list_bndbox = []
            path = os.path.join(folder_path_tuple, file_name)
            xml_save_path = os.path.join(xml_save_dir, file_name.replace(file_suffix, '.xml'))
            txt_path = os.path.join(txt_dir, file_name.replace(file_suffix, '.txt'))
            filename = os.path.splitext(file_name)[0]
            checked = 'NO'
            img = cv2.imread(path)
            height, width, depth = img.shape
            # print(height, width, depth)
            # im = Image.open(path)
            # im_w = im.size[0]
            # im_h = im.size[1]
            # width = str(im_w)
            # height = str(im_h)
            # depth = '3'
            flag = 'rectangle'
            pose = 'Unspecified'
            truncated = '0'
            difficult = '0'
            list_top.extend([xml_save_path, folder_path_tuple, filename, path, checked,
                             width, height, depth])
            try:
                for line in open(txt_path, 'r'):
                    line = line.strip()
                    info = line.split(' ')
                    name = dict[info[0]]
                    x_cen = float(info[1]) * width
                    y_cen = float(info[2]) * height
                    w = float(info[3]) * width
                    h = float(info[4]) * height
                    xmin = int(x_cen - w / 2)
                    ymin = int(y_cen - h / 2)
                    xmax = int(x_cen + w / 2)
                    ymax = int(y_cen + h / 2)
                    list_bndbox.extend([name, pose, truncated, difficult,
                                        str(xmin), str(ymin), str(xmax), str(ymax)])
                Xml_make().txt_to_xml(list_top, list_bndbox)
                COUNT += 1
                print(COUNT, xml_save_path)
            except:
                continue


if __name__ == '__main__':
    source_path = r'/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/data/nuscenes/v1.0-mini/samples/ALL_Images'  # txt标注文件所对应的的图片
    xml_save_dir = r'/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/data/nuscenes/v1.0-mini/v1.0-mini/xml'  # 转换为xml标注文件的保存路径
    txt_dir = r'/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/data/nuscenes/v1.0-mini/v1.0-mini/txt/ALL_txt'  # 需要转换的txt标注文件
    txt_2_xml(source_path, xml_save_dir, txt_dir)
```

### d. 剔除不存在标签的图片
由于某些图片中没有给定的类别，所以其没有对应的xml标签，需要将其剔除
```
import xml.dom.minidom
import os

# 改为自己的目录
root_path = '/home/xingchen/Study/4D_GT/VoxelNeXt_pipeline/'
annotation_path = root_path + 'data/nuscenes/v1.0-mini/v1.0-mini/xml/'
img_path = root_path + 'data/nuscenes/v1.0-mini/samples/ALL_Images/'
annotation_list = os.listdir(annotation_path)
img_list = os.listdir(img_path)

# for _ in annotation_list:
#     xml_name = _.split('.')[0]
#     img_name = xml_name + '.jpg'
#     if img_name not in img_list:
#         print("error xml:", img_name)
# print('ok')

if len(img_list) != len(annotation_list):
    print("图片和标签数目不匹配")
    if len(img_list) < len(annotation_list):
        print("标签比图片多")
        error_xml = []
        for _ in annotation_list:
            xml_name = _.split('.')[0]
            img_name = xml_name + '.jpg'
            if img_name not in img_list:
                error_xml.append(_)
                os.remove(annotation_path+_)
        print("error xml:", error_xml)
    else:
        print("图片比标签多")
        error_img = []
        for _ in img_list:
            img_name = _.split('.')[0]
            xml_name = img_name + '.xml'
            if xml_name not in annotation_list:
                error_img.append(_)
                os.remove(img_path+_)
        print("缺少标签的图片:", error_img)
```
剔除后检查，图片和xml标签文件数量均为2177

## 三、使用NuScenes进行训练
将NuScenes数据集的图片和xml标签软连接到VOCdevkit/VOC2007下

在model_data下新建nuScenes.txt存放类别
```
car
person
bicycle
```
修改voc_annotation.py中23行，train.py中77行中
```
classes_path        = 'model_data/voc5.txt'
```
为
```
classes_path        = 'model_data/nuScenes.txt'
```
修改yolo.py中30行
```
"classes_path"      : 'model_data/voc5.txt',
```
为
```
"classes_path"      : 'model_data/nuScenes.txt',
```
总训练轮次设定为100，其中冻结主干轮次为10，训练得到best_epoch_weights.pth

进行评估，修改yolo.py中28行
```
"model_path"        : 'model_data/yolov8_s.pth',
```
为
```
"model_path"        : 'logs/log3/best_epoch_weights.pth',
```
运行get_map.py，得到

![2024-01-04 16-09-16屏幕截图](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/d9b98b08-78e3-495e-bc75-055c6a81c13c)

运行predict.py，获得可视化结果

![n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151609937558](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/89c44822-272c-45c1-82c3-6668d9c40ccc)

![n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151609512404](https://github.com/xingchenshanyao/VoxelNeXt/assets/116085226/b626a7b8-09b2-4852-b24d-e07491a7f675)
