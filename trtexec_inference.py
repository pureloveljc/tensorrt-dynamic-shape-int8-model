import sys
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import copy
import numpy as np
import os
import torch
import cv2

import sys
import time
import copy
import os
import torch
import cv2
import torchvision.transforms as transforms
import copy
import time
from shapely.geometry import Polygon
import pyclipper


from ctypes import cdll, c_char_p

libcudart = cdll.LoadLibrary('/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p


def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + error_string)
cudaSetDevice(2)


class SegDetectorRepresenter():
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, batch, pred, is_output_polygon=False):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, H, W)
        '''
        # pred = pred[:, 0, :, :]
        # print(pred.shape)
        # 语义分割后图
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        # print(pred.shape[0])
        for batch_index in range(pred.shape[0]):
            # print(batch['img'][batch_index].shape)
            # height, width = batch['img'][batch_index].shape[1:]
            height, width = batch['shape'][batch_index]
            # print(batch['img'][batch_index].shape)
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.thresh

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()  # The first channel
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        # bitmap = _bitmap.cpu().numpy()  # The first channel
        # pred = pred.cpu().detach().numpy()
        height, width = _bitmap.shape
        contours, _ = cv2.findContours((_bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

def load_test_case(pagelocked_buffer, img):
    copy_size = img.ravel().size
    np.copyto(pagelocked_buffer[:int(copy_size)], img.ravel())
# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()
# Allocates all buffers required for an engine, i.e. host/device inputs/outputs

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    print('max_batch_size', engine.max_batch_size)
    for binding in engine:
        print('binding', binding, engine.get_binding_shape(binding),engine.get_binding_dtype(binding))
        size = 20000000
        # print(size)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

engine_file = '/mnt/stg3/dragon/ljc/TensorRT-8.2.4.2/bin/demo.trt'
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
a = (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
device = 'cuda:0'
with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    trt_engine = runtime.deserialize_cuda_engine(f.read())
    inputs, outputs, bindings, stream = allocate_buffers(trt_engine)

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def draw_box_on_img(img, text_polys):
    img_copy = copy.deepcopy(img)
    text_polys_copy = copy.deepcopy(text_polys)
    for box in text_polys_copy:
        box_reshape = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [box_reshape], True, (255, 0, 0), 2)
    # img_copy = cv2.resize(img_copy, (600, 800))
    cv2.imshow("image_before", cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def test_tensorrt(engine, test_loader):
    count = []
    with engine.create_execution_context() as context:
        context.set_optimization_profile_async(0, stream.handle)
        for path in os.listdir(test_loader):

            img = cv2.imread(os.path.join(test_loader, path), cv2.IMREAD_COLOR)
            img_rec = copy.deepcopy(img)
            (height, weight) = img.shape[:2]
            data = {"img": img}
            batch = {'shape': [(height, weight)]}
            data = resize_image(data=data)
            img_tensor = image_transformer(data["img"])
            img_tensor = torch.unsqueeze(img_tensor, dim=0).numpy().astype(np.float32)  # 扩展batch的维度
            data = img_tensor
            input_shape = engine.get_binding_shape(0)
            context.set_binding_shape(0, data.shape)
            if not context.all_binding_shapes_specified:
                raise RuntimeError("Not all input dimensions are specified for the exeuction context")
            load_test_case(inputs[0].host, data)
            start =time.time()
            pred = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=data.shape[0])

            # results = np.array(pred[0])[:data.shape[2] * data.shape[3]].reshape(1, data.shape[2], data.shape[3])  # 40ms
            results = torch.as_tensor(pred[0])[:data.shape[2] * data.shape[3]].view(1, data.shape[2], data.shape[3]).numpy()

            is_output_polygon = False
            # print(results.shape)
            box_list, score_list = post_processing(batch, results, is_output_polygon=is_output_polygon)

            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            t = time.time() - start
            count.append(t)
            print(t*1000)
            #
            # # print(len(pred))
            draw_box_on_img(img_rec, text_polys=box_list)
        average_data = my_AVERAGE_main(count)
        print("响应时间: ", (average_data * 1000, " ms"))
        del context

import time

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


def test_tensorrt_for_test(engine):
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    i = 0
    total_time_span = 0
    with engine.create_execution_context() as context:
        context.set_optimization_profile_async(0, stream.handle)
        input_shape = engine.get_binding_shape(0)
        input_shape[0] = engine.max_batch_size
        context.set_binding_shape(0,input_shape)
        if not context.all_binding_shapes_specified:
            raise RuntimeError("Not all input dimensions are specified for the exeuction context")
        # warm up
        print('input_shape', input_shape)
        data = np.random.rand(*input_shape).astype(np.float32)
        load_test_case(inputs[0].host, data)
        for i in range(10):
            pred = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=engine.max_batch_size)
        for i in range(100):
#             data = np.random.rand(*input_shape).astype(np.float32)
#             load_test_case(inputs[0].host, data)
            # =======================================
            # The common do_inference function will return a list of outputs - we only have one in this case.
            start_time = time.time()
            pred = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=engine.max_batch_size)
            time_span = time.time() - start_time
            total_time_span += time_span
        total_time_span /= 100.0
        print('total_time_span', total_time_span)
        # del context if not reuse
        del context


class ResizeShortSize:
    def __init__(self):
        """
        :param size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :return:
        """
        self.short_size = 512
        self.resize_text_polys = True

    def __call__(self, data: dict) -> dict:
        """
        对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        h, w = im.shape[:2]
        short_edge = min(h, w)
        if short_edge < self.short_size:
            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            scale = (scale, scale)
        else:
            if h < w:
                new_height = 1024
                new_width = new_height / h * w
            else:
                new_width = 1024
                new_height = new_width / w * h
            new_height = int(round(new_height / 32) * 32)
            new_width = int(round(new_width / 32) * 32)
            im = cv2.resize(im, (new_width, new_height))
        # data['img'] = im
        # data['img'] = cv2.resize(im, (1024, 1024))
        # logger.debug(data["img"].shape)
        return data


import torchvision.models as models
import torchvision.datasets as datasets
import tqdm
# change valdir to your imagenet dataset validation directory
resize_image = ResizeShortSize()
valdir = '/mnt/stg3/dragon/ljc/ocr_0407/jpg_test_batch/'
post_processing = SegDetectorRepresenter(
    thresh=0.3,
    box_thresh=0.7,
    max_candidates=1000,
    unclip_ratio=1.5
)
image_transformer = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
usinghalf = True

test_tensorrt(trt_engine, valdir)

# 响应时间:  (41.48529803413127, ' ms') trt   int
# 响应时间:  (54.41434951606413, ' ms') trt float
# 响应时间:  (128.36801241299435, ' ms')  predict