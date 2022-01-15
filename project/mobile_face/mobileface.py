import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

import pdb


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_planes, in_planes, kernel_size=kernel_size, padding=padding, groups=in_planes, bias=bias
        )
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class GDConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super(GDConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=in_planes, bias=bias
        )
        self.bn = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileFaceNet(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileFaceNet, self).__init__()
        block = InvertedResidual
        input_channel = 64
        last_channel = 512

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 5, 2],
                [4, 128, 1, 2],
                [2, 128, 6, 1],
                [4, 128, 1, 2],
                [2, 128, 2, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(inverted_residual_setting)
            )

        # building first layer
        # input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.conv1 = ConvBNReLU(3, input_channel, stride=2)
        self.dw_conv = DepthwiseSeparableConv(in_planes=64, out_planes=64, kernel_size=3, padding=1)
        features = list()
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.conv2 = ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        self.gdconv = GDConv(in_planes=512, out_planes=512, kernel_size=7, padding=0)
        self.conv3 = nn.Conv2d(512, 128, kernel_size=1)
        self.bn = nn.BatchNorm2d(128)
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x.size() -- torch.Size([2, 3, 112, 112])
        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.features(x)
        x = self.conv2(x)
        x = self.gdconv(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1)
        # pp x.size() -- torch.Size([2, 128])
        return x


def get_backbone():
    """Create model."""
    cdir = os.path.dirname(__file__)
    checkpoint = "models/mobile_face.pth" if cdir == "" else cdir + "/models/mobile_face.pth"

    model = MobileFaceNet()
    model.load_state_dict(torch.load(checkpoint))
    model = model.eval()

    return model


class Extractor(object):
    """Mobile face feature extractor"""

    def __init__(self, device=torch.device("cuda")):
        self.device = device
        self.backbone = get_backbone().to(device)
        self.transform = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, input_tensor):
        """input_tensor must be BxCxHxW and H, W must be 112, element range is [0, 1.0]"""
        assert input_tensor.size(2) == 112 and input_tensor.size(3) == 112

        for i in range(input_tensor.size(0)):
            input_tensor[i] = self.transform(input_tensor[i])

        with torch.no_grad():
            output_tensor_0 = self.backbone(input_tensor)
            output_tensor_1 = self.backbone(torch.flip(input_tensor, [3]))

        output_tensor = output_tensor_0 + output_tensor_1
        return output_tensor / torch.norm(output_tensor)

    def verify(self, f1, f2):
        LFW_SAME_FACE_THRESHOLD = 73.50
        cosine = torch.dot(f1, f2).clamp(-1.0, 1.0)
        theta = math.acos(cosine.item())
        theta = theta * 180 / math.pi
        is_same = theta < LFW_SAME_FACE_THRESHOLD
        return is_same, theta


if __name__ == "__main__":
    model = Extractor()

    input_tensor = torch.randn(2, 3, 112, 112).to(model.device)
    output_tensor = model(input_tensor)

    print(model.backbone)
    print("input_tensor: ", input_tensor.size())
    print("output_tensor: ", output_tensor.size())

    is_same, theta = model.verify(output_tensor[0], output_tensor[1])
    print(f"is same face ? {is_same}, theta: {theta}")
