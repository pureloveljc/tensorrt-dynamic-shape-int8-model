import torch
import torch.onnx
import onnxruntime

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Config import configs
from libs.tools import strLabelConverter
from libs.network.CRNN import get_crnn
from libs.dataset import alignCollate
from libs.dataset.tools import resizeNormalize
from loguru import logger
from abc import ABC
from PIL import Image
import numpy as np
import cv2
import argparse
import time
import glob


class OnnxModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''

        input_feed = self.get_input_feed(self.input_name, image_numpy)
        # scores = self.onnx_session.run(self.output_name[0], input_feed=input_feed)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class TestSet(Dataset, ABC):
    def __init__(self, args):
        super(TestSet, self).__init__()
        self.params = args
        self.input_h = self.params.MODEL.img_size.h
        self.input_w = self.params.MODEL.img_size.w
        self.mean = np.array(self.params.DATASET.mean, dtype=np.float32)
        self.std = np.array(self.params.DATASET.std, dtype=np.float32)
        self.data_dict = [img for i, img in enumerate(glob.glob(self.params.INFER.data_dir))][0:10000]
        logger.debug(f"test data dir: {self.params.INFER.data_dir}")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        img = cv2.imdecode(np.fromfile(self.data_dict[item], dtype=np.uint8), -1)
        img = resizeNormalize((self.input_w, self.input_h))(img)

        return img, item

def my_AVERAGE_main(data_list):

    if len(data_list) == 0:
        return 0
    if len(data_list) > 2:
        data_list.remove(min(data_list))
        data_list.remove(max(data_list))
        average_data = float(sum(data_list)) / len(data_list)
        return average_data
    elif len(data_list) <= 2:
        average_data = float(sum(data_list)) / len(data_list)
        return average_data

onnx_model_path = "./onnx_32_280.onnx"
model = OnnxModel(onnx_model_path)

parser = argparse.ArgumentParser(description='set path')
parser.add_argument('--cfg', default=f'/mnt/stg3/dragon/ljc/code/LightNet_v1.1.0/utils/ownData_config.yaml')
args = parser.parse_args()
params = configs(args)
test_data = TestSet(params)
logger.debug(f"test data size : {len(test_data.data_dict)}")

test_kwargs = {'pin_memory': False,
               'collate_fn': alignCollate(img_h=params.MODEL.img_size.h,
                                          img_w=params.MODEL.img_size.w)} if torch.cuda.is_available() else {}

test_loader = DataLoader(test_data,
                         batch_size=params.INFER.batch_size,
                         shuffle=False,
                         drop_last=False,
                         **test_kwargs)

# input_image, input_shape, origin_shape, batch, img_rec = preprocess_image(imagepath)
# pad_img, (new_height, new_width), (origin_height, origin_width)
# print(input_shape)
# segment_inputs, segment_outputs, segment_bindings = allocate_buffers(engine, True, input_shape)
begin_time = time.time()
sum_preds = []
sum_scores = []
time_all = []
convert = strLabelConverter(params.DATASET.ALPHABETS)
# start = time.time()
for i, (img, _) in enumerate(test_loader):
    # print(img)
    # print(img.size())
    start = time.time()
    img_ny = img.numpy()
    # img_ny = to_numpy(img_ny)

    preds = model.forward(img_ny)  # (67, 1, 7607)
    # model.forward(to_numpy(x))

    # preds = np.array(preds).reshape([67, img.size()[0], 7607])
    preds = torch.as_tensor(preds[0]).view(67, img_ny.shape[0], 7607)

    batch_size = img.size(0)
    preds_size = torch.IntTensor([preds.size(0)] * batch_size)

    score = preds
    _, preds = preds.max(2)
    preds_max = preds
    preds = preds.transpose(1, 0).contiguous().view(-1)
    # sim_preds, sum_score = self.convert.decode(preds.data, preds_size.data,score=score, pred_max=preds_max, raw=False)
    # sum_preds += sim_preds
    # sum_scores += sum_score
    sim_preds = convert.decode(preds.data, preds_size.data,score=score, pred_max=preds_max, raw=False)
    sum_preds += sim_preds
    t = time.time() - start
    time_all.append(t)
    print(t*1000)
    # end = time.time()

#     print((end-start)*1000)
# end = time.time()
time_result = my_AVERAGE_main(time_all)
print('result: {}\n{}\n, spend time: {}  s'.format(sum_preds, sum_scores,  time_result*1000))

# x = torch.rand(1, 3, 32, 280)
# out = model.forward(to_numpy(x))
# print(out)

# [0:5000]
# , spend time: 0.05511728130352842 s   predict
# , spend time: 0.046901512759291476  s  onnx


# , spend time: 0.023580137454870233  s   onnx
# , spend time: 0.22425426679093804  s
# , spend time: 0.02836601883412173 s    predict

#  batch 1   spend time: 0.024785574948127137 s   24.7ms
#  batch 1   spend time: 0.01060946787994226 s   10.7ms


#  batch 1   spend time: 0.024785574948127137 s   24.7ms
#  batch 1   spend time: 0.01060946787994226 s   10.7ms