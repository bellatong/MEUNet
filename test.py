#!/usr/bin/python3
# coding=utf-8

import os
import sys

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

import torch

from skimage import io
from PIL import Image
from torch.utils.data import DataLoader

import dataset
from MEUNet import MEUNet


def normPRED(d):
    ma = torch.max(d+1)
    mi = torch.min(d+1)

    dn = (d+1 - mi) / (ma - mi)

    return dn

def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


class Test(object):
    def __init__(self, Dataset, Network, Path,image_file_name):
        ## dataset
        self.model_name = 'U3Net_epoch_596_itr_394020_train_0.090393'
        self.cfg = Dataset.Config(datapath=Path,image_file_name=image_file_name, snapshot='./out/'+self.model_name+'.pth', mode='test')
        self.net = Network(self.cfg)
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network

        self.net.train(False)
        self.net.cuda()


    def save_prediction(self,dataset_name):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape = image.cuda().float(), (H, W)

                print("inferencing:%s"%name)
                d_edge, d_united, d1, d2, d3, d4 = self.net(image)

                # add
                # pred = torch.sigmoid(d_united[0,0])
                pred = torch.sigmoid(d_united[0,0])

                #can be canceled
                pred = normPRED(pred)

                pred = pred.cpu().numpy() * 255
                #
                head = './save_datas/' + self.model_name+'/'+dataset_name+ os.sep
                if not os.path.exists(head):
                    os.makedirs(head, exist_ok=True)
                cv2.imwrite(head +  name[0] + '.png', np.round(pred))

                # edge prediction
                # pred = torch.sigmoid(d_edge[0, 0]).cpu().numpy() * 255
                # head = './save_datas/' +self.model_name+'/edge/'+dataset_name+ os.sep
                # if not os.path.exists(head):
                #     os.makedirs(head, exist_ok=True)
                # cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))




if __name__ == '__main__':
    for dataset_name in ['DUTS-TE', 'DUT-OMROM', 'ECSSD', 'HKU-IS', 'PASCAL-S', 'SOD']:
        image_root_dir = "/home/bianyetong/datasets"
        # image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
        # prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
        # model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
        if dataset_name == 'DUTS-TE':
            image_dir = os.path.join(image_root_dir, 'DUTS/DUTS-TE')
            image_file_name = '/Image'
        elif dataset_name == 'DUT-OMROM':
            image_dir = os.path.join(image_root_dir, 'DUT-OMROM')
            image_file_name = '/Image'
        elif dataset_name == 'ECSSD':
            image_dir = os.path.join(image_root_dir, 'ECSSD')
            image_file_name = '/Image'
        elif dataset_name == 'HKU-IS':
            image_dir = os.path.join(image_root_dir, 'HKU-IS')
            image_file_name = '/Img'
        elif dataset_name == 'PASCAL-S':
            image_dir = os.path.join(image_root_dir, 'PASCAL-S')
            image_file_name = '/Imgs'

        elif dataset_name == 'SOD':
            image_dir = os.path.join(image_root_dir, 'SOD')
            image_file_name = '/Imgs/Imgs'

        t = Test(dataset, MEUNet, image_dir, image_file_name)
        t.save_prediction(dataset_name)
