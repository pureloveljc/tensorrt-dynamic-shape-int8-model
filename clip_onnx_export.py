
import clip
import torch
from torch import nn
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

device = "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)  # Load any model
model = model.eval() # Inference Only

img_size = model.visual.input_resolution
print(img_size)
print(model)
dummy_image = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)
image_embedding = model.encode_image(dummy_image).to(device)
print(dummy_image)


class Mycls(nn.Module):
    def __init__(self, ori_net):
        super(Mycls, self).__init__()
        self.ori_net = ori_net

    def forward(self, x):
        fea = self.ori_net.encode_image(x)

        # print(scores)
        # print(index)


        return fea
print()
model_ft = Mycls(model)
model_ft = model_ft.to(device)
model_ft = model_ft.eval()
with torch.no_grad():
    y_onnx = torch.onnx._export(model_ft,
                                dummy_image,
                                "clip_ViT-B_32.onnx",
                                export_params=True,  # ????????????????
                                opset_version=11,
                                input_names=["input"],  # ????????????????????????????????key????
                                output_names=['output'],
                                dynamic_axes={'input': {0: 'batchsize'},
                                              'output': {0: 'batchsize'}},
                                verbose=True,
                                do_constant_folding=True
                                )

with torch.no_grad():
    y = model_ft(dummy_image)
    print(y)
    print('~~~~~~~~~')
    print(y_onnx)
    print("result", torch.max(torch.abs(y - y_onnx)))
