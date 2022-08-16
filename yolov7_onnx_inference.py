import cv2
import time
import numpy as np
import onnxruntime
from PIL import Image
import random
det_model = "/mnt/stg3/dragon/ljc/code/yolov7/runs/train/yolov7-w6_lcaaa10/weights/best.onnx"
sess_det = onnxruntime.InferenceSession(det_model, providers=['CUDAExecutionProvider'])

# def init_engine(self):
#     providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device else ['CPUExecutionProvider']
#     self.session = ort.InferenceSession(self.weights, providers=providers)
def predict(im, session):
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: im}
    outputs = session.run(outname, inp)[0]
    return outputs

def preprocess(image_path, session):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img.copy()
    im, ratio, dwdh = letterbox(image, auto=False)
    t1 = time.time()
    outputs = predict(im, session)
    print("锟斤拷锟斤拷锟斤拷锟斤拷", (time.time() - t1) * 1000, ' ms')
    ori_images = [img.copy()]
    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0, y0, x1, y1])
        box -= np.array(dwdh * 2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 3)
        name = names[cls_id]
        color = colors[name]
        name += ' ' + str(score)
        cv2.rectangle(image, box[:2], box[2:], color, 2)
        cv2.putText(image, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
    a = Image.fromarray(ori_images[0])
    return a

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # cv2.imshow('src', img)
    # cv2.waitKey(0)
    im = img.transpose((2, 0, 1))
    im = np.expand_dims(im, 0)
    im = np.ascontiguousarray(im)
    im = im.astype(np.float32)
    im /= 255
    return im, ratio, (dw, dh)



img = cv2.imread('/mnt/stg3/dragon/ljc/code/yolov7/863a0862-c52a-4a48-a107-1764e29fd9c5-1660626896047.jpg')
raw_h, raw_w, c = img.shape

print(img.shape)
img_new, ratio, dwdh = letterbox(img, (1280, 1280), auto=False)
print(img_new.shape)
for i in range(500):
    t1 = time.time()
    dets = predict(img_new, sess_det)


    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img_rgb.copy()
    # im, ratio, dwdh = letterbox(image, auto=False)
    # t1 = time.time()
    # outputs = predict(im, session)
    # print("锟斤拷锟斤拷锟斤拷锟斤拷", (time.time() - t1) * 1000, ' ms')
    ori_images = [img_rgb.copy()]
    names = ["icon", "text"]
    color = [random.randint(0, 255) for _ in range(3)]
    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(dets):
        image = ori_images[int(batch_id)]

        box = np.array([x0, y0, x1, y1])
        box -= np.array(dwdh * 2)
        # box /= ratio
        box = np.array([box[0]/ratio[0], box[1]/ratio[1], box[2]/ratio[0], box[3]/ratio[1]])
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 3)
        name = names[cls_id]
        # color = color[cls_id]
        name += ' ' + str(score)
        # print(box)
        # print((box[:2]))
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
        cv2.putText(image, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [0, 255, 255], thickness=2)
    # print("time:", (time.time() - t1) * 1000, ' ms')
    image = cv2.resize(image, (800, 800))
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('src', img_rgb)
    cv2.waitKey(0)
# a = Image.fromarray(ori_images[0])


# onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Concat node. Name:'Concat_203' Status Message:
# concat.cc:156 PrepareForCompute Non concat axis dimensions must match: Axis 2 has mismatched dimensions of 24 and 23
