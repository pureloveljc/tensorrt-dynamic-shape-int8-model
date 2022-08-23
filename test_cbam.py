import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       #  print(x.size())
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class InceptionA(nn.Module):

    def __init__(self, in_channels):

        super(InceptionA,self).__init__()

        self.branch1_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5_5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5_5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.branch3_3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3_3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3_3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):

# x = self.branch1????1(x)

# x = self.branch5????5_1(x)

# x = self.branch5????5_2(x)

# ???????????????????????????????????????????????????????????????????????????? x????????????????????????????????????????????????????????????????????????????????

        branch1_1 = self.branch1_1(x)
        branch5_5 = self.branch5_5_1(x)
        branch5_5 = self.branch5_5_2(branch5_5)
        branch3_3 = self.branch3_3_1(x)
        branch3_3 = self.branch3_3_2(branch3_3)
        branch3_3 = self.branch3_3_3(branch3_3)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1_1, branch5_5, branch3_3, branch_pool]
        return torch.cat(outputs, dim=1) # (b, c, w, h),????dim=1????????????????????????????????????????????


if __name__ == "__main__":

    cbam_custom_64 = CBAM(64)
    cbam_custom_128 = CBAM(128)
    cbam_custom_256 = CBAM(256)
    cbam_custom_512 = CBAM(512)


    resnet18 = models.resnet18(pretrained=True)
    model_copy = copy.deepcopy(resnet18)
    resnet18.layer1 = nn.Sequential(resnet18.layer1[0], cbam_custom_64, resnet18.layer1[1], cbam_custom_64)
    resnet18.layer2 = nn.Sequential(resnet18.layer2[0], cbam_custom_128, resnet18.layer2[1], cbam_custom_128)
    resnet18.layer3 = nn.Sequential(resnet18.layer3[0], cbam_custom_256, resnet18.layer3[1], cbam_custom_256)
    resnet18.layer4 = nn.Sequential(resnet18.layer4[0], cbam_custom_512, resnet18.layer4[1], cbam_custom_512)

    print(resnet18)
