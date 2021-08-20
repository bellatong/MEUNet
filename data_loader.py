#!/usr/bin/python3
# coding = utf-8

import torch
import numpy as np
import random
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, mask1, mask2, mask3, mask4, mask5):
        image = (image - self.mean) / self.std
        if mask is None:
            return image
        return image, mask / 255, mask1 / 255, mask2 / 255, mask3 / 255, mask4 / 255, mask5 / 255


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label,edge = sample['imidx'], sample['image'], sample['label'],sample['edge']

        if random.random() >= 0.5:
            image = image[::-1].copy()
            label = label[::-1].copy()
            edge = edge[::-1].copy()

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]
        edge = edge[top: top + new_h, left: left + new_w]

        return {'imidx': imidx, 'image': image, 'label': label,'edge':edge}


class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))  # 为什么输入需要是（int，tuple）
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label,edge = sample['imidx'], sample['image'], sample['label'],sample['edge']

        # if random.random() >= 0.5:
        #     image = image[::-1].copy()
        #     label = label[::-1].copy()
        #     edge = edge[::-1].copy()

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                # output_size是指rescale之后更短的那一条
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        # transform.resize()剪裁后的图片是以float64的格式存储的，数值的取值范围是（0～1）
        # 如果通过cv2.resize()剪裁，剪裁之后的图片还是以numpy array形式保存
        img = transform.resize(image, (new_h, new_w), mode='constant')
        lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)
        edg = transform.resize(edge,(new_h, new_w), mode='constant', order=0, preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl,'edge':edg}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        imidx, image, label, edge = sample['imidx'], sample['image'], sample['label'], sample['edge']

        # 用0填充出一个image.shape[0]×image.shape[1]×3大小的array
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)
        tmpEdg = np.zeros((edge.shape[0],edge.shape[1],1))

        image = image / np.max(image)
        # 这一步把像素值都映射到0-1
        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        if (np.max(edge) < 1e-6):
            edge = edge
        else:
            edge = edge / np.max(edge)

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]
        tmpEdg[:,:,0] = edge[:,:,0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))
        tmpEdg = edge.transpose((2,0,1))

        return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl),'edge':torch.from_numpy(tmpEdg)}


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, edg_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.edge_name_list = edg_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = io.imread(self.image_name_list[idx])
        imname = self.image_name_list[idx]
        imidx = np.array([idx])

        if (0 == len(self.label_name_list)):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        label = np.zeros(label_3.shape[0:2])
        if (3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif (2 == len(label_3.shape)):
            label = label_3

        if (3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
            image = image[:, :, np.newaxis]

        if (0 == len(self.edge_name_list)):
            edge_3 = np.zeros(image.shape)
        else:
            edge_3 = io.imread(self.edge_name_list[idx])

        edge = np.zeros(edge_3.shape[0:2])
        if (3 == len(edge_3.shape)):
            edge = edge_3[:, :, 0]
        elif (2 == len(edge_3.shape)):
            edge = edge_3

        if (3 == len(image.shape) and 2 == len(edge.shape)):
            edge = edge[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(edge.shape)):
            edge = edge[:, :, np.newaxis]
            #image = image[:, :, np.newaxis]

        sample = {'imidx': imidx, 'image': image, 'label': label, 'edge': edge}

        if self.transform:
            sample = self.transform(sample)

        return sample
