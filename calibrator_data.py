import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
import torch
import pycuda.autoinit
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torchvision.transforms as transforms
from ctypes import cdll, c_char_p

libcudart = cdll.LoadLibrary('/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p


def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + str(error_string))


# cudaSetDevice(1)

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
        # im = cv2.resize(im, (1024, 1024))
        data['img'] = im
        # data['img'] = cv2.resize(im, (768, 1280))
        # logger.debug(data["img"].shape)
        return data

resize_image = ResizeShortSize()
image_transformer = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

class dbEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dir, cache_file):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        # self.model_shape = model_shape
        self.num_calib_imgs = 484  # the number of images from the dataset to use for calibration
        self.batch_size = 1
        self.batch_shape = (5000, 5000, 3)
        self.cache_file = cache_file

        calib_imgs = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.calib_imgs = np.random.choice(calib_imgs, self.num_calib_imgs)
        # print(self.calib_imgs )
        self.counter = 0 # for keeping track of how many files we have read
        self.device_input = cuda.mem_alloc(trt.volume(self.batch_shape) * trt.float32.itemsize)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):

        # if there are not enough calibration images to form a batch,
        # we have reached the end of our data set
        if self.counter == self.num_calib_imgs:
            return None
        # batch_imgs = []
        # try:
        # batch_imgs = np.zeros((self.batch_size, 1 * 3 * h * w * self.batch_size))
        try:
            for i in range(self.batch_size):
                print(self.calib_imgs[self.counter + i])
                img = cv2.imread(self.calib_imgs[self.counter + i], cv2.IMREAD_COLOR)
                data = {"img": img}
                height, weight = img.shape[:2]
                data = resize_image(data=data)
                img = image_transformer(data["img"]).numpy()
                # img = img.transpose((1, 2, 0))
                print(img.shape)
                if img.all():
                    h, w = img.shape[1:]
                    batch_imgs = np.zeros((self.batch_size, trt.volume(self.batch_shape)))
                    batch_imgs[i, :h*w*3] = img.ravel()
                else:
                    continue
            self.counter += self.batch_size
            # Copy to device, then return a list containing pointers to input device buffers.
            # cuda.memcpy_htod(self.device_input)
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))
            return [int(self.device_input)]
        except Exception as e:
            print(e)
            print('except')
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


if __name__ == '__main__':

    dbEntropyCalibrator("/mnt/stg3/dragon/ljc/ocr_0401/jpg/", "./ocr_cali.cache")