import sys

from torch import nn
from torchvision import models
from pretrainedmodels import se_resnext101_32x4d, se_resnext50_32x4d, senet154
from pretrainedmodels import inceptionresnetv2

sys.path.append("/usr/src/app/Kaggle_Bengali2019")
from src.layers import AdaptiveConcatPool2d, Flatten, SEBlock, GeM, CBAM_Module


encoders = {
    "se_resnext50_32x4d": {
        "encoder": se_resnext50_32x4d,
        "out_shape": 2048
    },
    "se_resnext101_32x4d": {
        "encoder": se_resnext101_32x4d,
        "out_shape": 2048
    },
    "inceptionresnetv2": {
        "encoder": inceptionresnetv2,
        "out_shape": 1536
    },
    "resnet34": {
        "encoder": models.resnet34,
        "out_shape": 512
    },
    "resnet50": {
        "encoder": models.resnet50,
        "out_shape": 2048
    },
    "resnet50_cbam": {
        "encoder": models.resnet50,
        "layer_shapes": [2048, 1024, 512, 256, 64],
        "out_shape": 2048
    }
}


class CnnModel(nn.Module):
    def __init__(self, num_classes, encoder="se_resnext50_32x4d", pretrained="imagenet", pool_type="concat"):
        super().__init__()
        self.net = encoders[encoder]["encoder"](pretrained=pretrained)

        if encoder == "resnet50_cbam":
            self.net.layer1 = nn.Sequential(self.net.layer1,
                                         CBAM_Module(encoders[encoder]["layer_shapes"][3]))
            self.net.layer2 = nn.Sequential(self.net.layer2,
                                         CBAM_Module(encoders[encoder]["layer_shapes"][2]))
            self.net.layer3 = nn.Sequential(self.net.layer3,
                                         CBAM_Module(encoders[encoder]["layer_shapes"][1]))
            self.net.layer4 = nn.Sequential(self.net.layer4,
                                         CBAM_Module(encoders[encoder]["layer_shapes"][0]))

        if encoder in ["resnet34", "resnet50", "resnet50_cbam"]:
            if pool_type == "concat":
                self.net.avgpool = AdaptiveConcatPool2d()
                out_shape = encoders[encoder]["out_shape"] * 2
            elif pool_type == "avg":
                self.net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                out_shape = encoders[encoder]["out_shape"]
            elif pool_type == "gem":
                self.net.avgpool = GeM()
                out_shape = encoders[encoder]["out_shape"]
            self.net.fc = nn.Sequential(
                Flatten(),
                SEBlock(out_shape),
                nn.Dropout(),
                nn.Linear(out_shape, num_classes)
            )
        elif encoder == "inceptionresnetv2":
            if pool_type == "concat":
                self.net.avgpool_1a = AdaptiveConcatPool2d()
                out_shape = encoders[encoder]["out_shape"] * 2
            elif pool_type == "avg":
                self.net.avgpool_1a = nn.AdaptiveAvgPool2d((1, 1))
                out_shape = encoders[encoder]["out_shape"]
            elif pool_type == "gem":
                self.net.avgpool_1a = GeM()
                out_shape = encoders[encoder]["out_shape"]
            self.net.last_linear = nn.Sequential(
                Flatten(),
                SEBlock(out_shape),
                nn.Dropout(),
                nn.Linear(out_shape, num_classes)
            )
        else:
            if pool_type == "concat":
                self.net.avg_pool = AdaptiveConcatPool2d()
                out_shape = encoders[encoder]["out_shape"] * 2
            elif pool_type == "avg":
                self.net.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                out_shape = encoders[encoder]["out_shape"]
            elif pool_type == "gem":
                self.net.avg_pool = GeM()
                out_shape = encoders[encoder]["out_shape"]
            self.net.last_linear = nn.Sequential(
                Flatten(),
                SEBlock(out_shape),
                nn.Dropout(),
                nn.Linear(out_shape, num_classes)
            )


    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.net(x)
