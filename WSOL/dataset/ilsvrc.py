import os
import numpy as np
from PIL import Image
import paddle
import paddle.vision.transforms as transforms
from paddle.io import Dataset, DataLoader
from xml.etree import ElementTree as ET

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            lineSplit = line.strip().split(' ')
            imgPath, label = lineSplit[0], lineSplit[1]
            flag = lineSplit[2]
            imgList.append((imgPath, int(label), str(flag)))

    return imgList

def bboxes_reader(path):
    bboxes_list = {}
    bboxes_file = open(path + "/val.txt")
    for line in bboxes_file:
        line = line.split('\n')[0]
        line = line.split(' ')[0]
        labelIndex = line
        line = line.split("/")[-1]
        line = line.split(".")[0] + ".xml"
        bbox_path = path + "/val_boxes/val/" + line
        tree = ET.ElementTree(file=bbox_path)
        root = tree.getroot()
        ObjectSet = root.findall('object')
        bbox_line = []
        for Object in ObjectSet:
            BndBox = Object.find('bndbox')
            xmin = BndBox.find('xmin').text
            ymin = BndBox.find('ymin').text
            xmax = BndBox.find('xmax').text
            ymax = BndBox.find('ymax').text
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            bbox_line.append([xmin, ymin, xmax, ymax])
        bboxes_list[labelIndex] = bbox_line
    return bboxes_list

class ILSVRC2012(Dataset):
    def __init__(self, root, input_size, crop_size, train=True, transform=None):
        self._root = root
        self._train = train
        self._input_size = input_size
        self._crop_size = crop_size
        self._transform = transform
        self.loader = pil_loader

        if self._train:
            self.imgList = default_list_reader(self._root + '/train.txt')#[:1000]
        else:
            self.imgList = default_list_reader(self._root + '/val.txt')#[:1000]

        self.bboxes = bboxes_reader(self._root)

    def __getitem__(self, index):
        img_name, label, cls_name = self.imgList[index]

        image = self.loader(os.path.join(self._root, img_name))

        newBboxes = []
        if not self._train:
            bboxes = self.bboxes[img_name]
            for bbox_i in range(len(bboxes)):
                bbox = bboxes[bbox_i]
                bbox[0] = bbox[0] * (self._input_size / image.size[0]) - (self._input_size - self._crop_size) / 2
                bbox[1] = bbox[1] * (self._input_size / image.size[1]) - (self._input_size - self._crop_size) / 2
                bbox[2] = bbox[2] * (self._input_size / image.size[0]) - (self._input_size - self._crop_size) / 2
                bbox[3] = bbox[3] * (self._input_size / image.size[1]) - (self._input_size - self._crop_size) / 2
                bbox.insert(0, index)
                newBboxes.append(bbox)

        if self._transform is not None:
            image = self._transform(image)

        if self._train:
            return image, label, cls_name, img_name.split('/')[-1].split('.')[0]
        else:
            return image, label, newBboxes, cls_name, img_name.split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.imgList)

def my_collate(batch):
    images = []
    labels = []
    bboxes = []
    cls_name = []
    img_name = []
    for sample in batch:
        images.append(sample[0])
        labels.append(np.array(sample[1]))  # Convert label to numpy array
        bboxes.append(np.array(sample[2]))  # Convert bbox to numpy array
        cls_name.append(sample[3])
        img_name.append(sample[4])

    return paddle.stack(images, 0), paddle.to_tensor(labels), bboxes, cls_name, img_name

if __name__ == '__main__':
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_data = ILSVRC2012(root='/data1/xjheng/dataset/ILSVRC2012', input_size=256, crop_size=224, train=False, transform=test_transforms)
    print(len(test_data))

    test_loader = DataLoader(
        test_data, batch_size=2, shuffle=False,
        num_workers=0, collate_fn=my_collate)

    for i, (input, target, boxes, cls_name, img_name) in enumerate(test_loader):
        print(cls_name, img_name)
        print(boxes)
        gtboxes = [boxes[k][:, 1:] for k in range(len(boxes))]
        print(gtboxes)
        break
