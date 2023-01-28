import cv2
import numpy as np
import onnxruntime
import os
import torch
from torch import nn
import torch.optim as optim
import torchvision
#pip install torchvision
from torchvision import transforms, models, datasets
#https://pytorch.org/docs/stable/torchvision/index.html
import time
import warnings
import random
import sys
import copy
import json
import cv2
from PIL import Image
import torch.utils.data

import os
torch.set_printoptions(precision=20, sci_mode=False)

classes_names =  ['other', 'name1', 'name2', 'name3']


class IconCls():

    def __init__(self, model_path):

        self.sess = onnxruntime.InferenceSession(
            model_path, providers=['CUDAExecutionProvider'])
        self.width = 224
        self.height = 224

    def resize_norm_img(self, img_path):

        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.width, self.height))

        std = np.array([0.229, 0.224, 0.225]).reshape(
            (1, 1, 3)).astype('float32')
        mean = np.array([0.485, 0.456, 0.406]).reshape(
            (1, 1, 3)).astype('float32')
        img = (img.astype(np.float32)/255 - mean) / std
        img = img.transpose((2, 0, 1))
        img = img[np.newaxis, :]
        return img




    def infer(self, img_paths):
        batch_img = []

        dst_img = self.resize_norm_img(img_paths)
        batch_img.append(dst_img)
        # if not batch_img:
        #     return []
        batch_img = np.concatenate(batch_img)
        # img = img.numpy()
        pred = self.sess.run(['output'], {'input': batch_img})[0]
        # results = [class_names[np.argmax(self.softmax(pred[i, :]))]
        #           for i in range(pred.shape[0])]
        # results = [class_names[np.argmax(self.softmax(pred[i, :]))] if self.softmax(pred[i, :])[
        #     np.argmax(self.softmax(pred[i, :]))] > 0.85 else "????????" for i in range(pred.shape[0])]
        return pred


# ics = IconCls("/home/ljc/pytorch_demo/1111_class153/easy1127_best.onnx")
ics = IconCls("/home/all/ljc/code/ConvNeXt-V2/convnext.onnx")
# /home/ljc/pytorch_demo/1111_class153/easy1129_aaa.onnx


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def raw():  # ????????
    result = []
    train_on_gpu = torch.cuda.is_available()
    root_path = ""
    all_nums = 0
    weight_acc= 0
    for class_name in os.listdir("/home/all/ljc/icon_0124/val/"):
        acc_count = 0
        all_count = 0
        # if class_name == "other":
        #     class_name = "????"
        for c, cls in enumerate(os.listdir(os.path.join("/home/all/ljc/icon_0124/val//", class_name))):
            all_count += 1
            image_path = os.path.join(os.path.join("/home/all/ljc/icon_0124/val//", class_name), cls)
            # image_path = "/home/ljc/small_jpg/????-1/00.png"
            all_nums += 1
            # print(image_path)
            # start = time.time()
            # img = process_image(image_path)
            # img = transform_aug(image_path)
            #img.transpose((2, 0, 1))j cl

            # print(image_path)
            import time
            try:
                a = time.time()
                pred = ics.infer(image_path)
                # print(pred)
                b = time.time()
                # print((b-a)*1000)
            except Exception as e:
                print(image_path)
                print(e)
                continue

            result1 = classes_names[int(pred[0, :][0])]
            print(result1)
            print(class_name)
            print('~~~~~~~~~')
            if result1 == class_name:
                acc_count += 1
        # print(all_count)
        # print(all_nums)

        # weight_acc += cls_acc
        result.append([class_name, acc_count/all_count, all_count])
        # print("class_name: {} acc :".format(class_name)+str(acc_count/all_count) + ", val nums:"+str(all_count))
    for i in range(len(result)-1):
        for j in range(len(result)-1-i):
            if result[j][1] > result[j+1][1]:
                result[j], result[j+1] = result[j+1], result[j]
    for i in result:
        print("class_name: {} acc :".format(i[0]) + str(i[1]) + ", val nums:" + str(i[2]))
        # cls_acc = (all_count/all_nums)*(acc_count/all_count)
    weight_acc = 0
    all_nums = 0
    for i in result:
        all_nums += (i[2])
    for i in result:
        weight_acc += (i[2]/all_nums)*i[1]
    print("acc:"+str(weight_acc))
    # imshow(img)
    # plt.show()

raw()

