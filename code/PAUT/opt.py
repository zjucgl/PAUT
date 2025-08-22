import torchvision.models as models

from code.PAUT.net import mbv2_ca
from code.PAUT.net import MobileNetV3_Small, MobileNetV3_Large
from code.PAUT.net import InceptionV4
from code.PAUT.net import resnet152_slice
from code.PAUT.net.ResNet_att import resnet152_att, resnet50_att, resnet18_att
from code.PAUT.net import eca_resnet152
from code.PAUT.net import Fuse_model
from code.PAUT.net import Fuse_PAUT
from code.PAUT.net import resnet18_fpn

import sys

Local = sys.platform.startswith("win")

# workspace_dir = 'E:/datasets/gydp/classification/datasets' if Local \
#     else '/home/admin/datasets/gydp/classification/datasets'
# cls = ['CR', 'LF', 'LP', 'ND', 'PO']
CUDA_INDEX = 1

ROOT = '/root/datadisk/hb/datasets'

workspace_dir = r'E:\datasets\datasets' if Local \
    else '/root/datadisk/hb/datasets/datasets-s-'

type_path = '/root/datadisk/hb/datasets/datasets-s-3'
cls = os.listdir(os.path.join(type_path, 'testing')) if not Local else ['CR', 'LF', 'LP', 'ND']

input_size = (224, 224)
FORMER_SLICE = False

base_res_dir = 'E:/tmp/runs' if Local else f'/root/datadisk/hb/paut/runs'

dropout = 0.0


def get_model_by_str(model_name, pretrained=False, pretrain_path='', freeze=False, att=None, drop_rate=0.):
    model = None

    if model_name in ['resnet18', 'resnet50', 'resnet152', 'resnet152_slice', 'resnet152_att', 'eca_resnet152',
                      'resnet50_att', 'resnet18_att', 'resnet18_fuse', 'resnet18_paut', 'resnet18_fpn']:
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        if model_name == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
        if model_name == 'resnet152_att':
            model = resnet152_att(att=att, drop_rate=drop_rate)
            if pretrained:
                model.load_state_dict(torch.load('./weights/resnet152_v1.pth'), strict=False)
        if model_name == 'resnet18_att':
            model = resnet18_att(att=att, drop_rate=drop_rate)
            if pretrained:
                model.load_state_dict(torch.load('weights/resnet18_v1.pth'), strict=False)
        if model_name == 'resnet50_att':
            model = resnet50_att(att=att, drop_rate=drop_rate)
            if pretrained:
                model.load_state_dict(torch.load('weights/resnet50_v1.pth'), strict=False)
        if model_name == 'resnet18_fuse':
            model = Fuse_model()
        if model_name == 'resnet18_paut':
            model = Fuse_PAUT()
        if model_name == 'resnet18_fpn':
            model = resnet18_fpn()
        if model_name == 'resnet152_slice':
            model = resnet152_slice()
            if pretrained:
                model.load_state_dict(torch.load('./weights/resnet152.pth'), strict=False)
        if model_name == 'eca_resnet152':
            model = eca_resnet152()
            if pretrained:
                model.load_state_dict(torch.load('./weights/eca_resnet152.pth'))
        # 冻结模型的所有参数
        if freeze:
            for name, param in model.named_parameters():
                if name.startswith('conv1') or name.startswith('layer1'):
                    param.requires_grad = False

        if model_name != 'resnet18_fpn':
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(cls))
    elif model_name in ['densenet121', 'densenet201', 'densenet121_my', 'densenet201_att']:
        if model_name == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
        if model_name == 'densenet201':
            model = models.densenet201(pretrained=pretrained)
        if model_name == 'densenet121_att':
            model = densenet121_att()
            if pretrained:
                state_dict = torch.load('./weights/densenet121_v1.pth')
                # 初始化一个空 dict
                new_state_dict = OrderedDict()
                # 修改 key
                for k, v in state_dict.items():
                    if 'denseblock' in k:
                        param = k.split(".")
                        k = ".".join(param[:-3] + [param[-3] + param[-2]] + [param[-1]])
                        print(k)
                    new_state_dict[k] = v
                model.load_state_dict(new_state_dict, strict=False)
                # num_features = model.classifier.in_features
                # # 定义新的分类器
                # model.classifier = nn.Linear(num_features, len(cls))
                # model.load_state_dict(torch.load(pretrain_path), strict=False)
        # 冻结模型的所有参数
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        # 修改分类器部分
        num_features = model.classifier.in_features
        # 定义新的分类器
        model.classifier = nn.Linear(num_features, len(cls))
    elif model_name in ['googlenet']:
        model = models.googlenet(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(cls))

    elif model_name in ['inceptionv4']:
        if model_name == 'inceptionv4':
            model = InceptionV4(num_classes=1001)
            model.load_state_dict(torch.load('./weights/inceptionv4.pth'))

            new_last_linear = nn.Linear(1536, len(cls))
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

            settings = {
                'input_space': 'RGB',
                'input_size': [3, 299, 299],
                'input_range': [0, 1],
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5]
            }
            model.input_space = settings['input_space']
            model.input_size = settings['input_size']
            model.input_range = settings['input_range']
            model.mean = settings['mean']
            model.std = settings['std']

            if pretrain_path:
                model.load_state_dict(torch.load(pretrain_path))

    elif model_name in ['mobilenet_v2']:
        if model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
        # 冻结模型的所有参数
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        # building classifier
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(model.last_channel, len(cls)),
        )
    elif model_name in ['mobilenet_v2_ca']:
        model = mbv2_ca()
        if pretrained:
            model.load_state_dict(
                torch.load("./weights/mbv2_ca.pth", map_location='cpu'), strict=False)
        model.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(model.last_channel, len(cls))
        )
    elif model_name in ['mobilenet_v3_small', 'mobilenet_v3_large']:
        if model_name == 'mobilenet_v3_small':
            model = MobileNetV3_Small()
            if pretrained:
                model.load_state_dict(
                    torch.load("./weights/mobilenetv3/450_act3_mobilenetv3_small.pth", map_location='cpu'))
        if model_name == 'mobilenet_v3_large':
            model = MobileNetV3_Large()
            if pretrained:
                model.load_state_dict(
                    torch.load("./weights/mobilenetv3/450_act3_mobilenetv3_large.pth", map_location='cpu'))
        model.linear4 = nn.Linear(1280, len(cls))
    elif model_name in ['squeezenet1_0', 'squeezenet1_1']:
        if model_name == 'squeezenet1_0':
            model = models.squeezenet1_0(pretrained=pretrained)
        if model_name == 'squeezenet1_1':
            model = models.squeezenet1_1(pretrained=pretrained)
        if freeze:
            # 冻结除最后一层外的所有参数
            for param in model.parameters():
                param.requires_grad = False

        final_conv = nn.Conv2d(512, len(cls), kernel_size=(1, 1))
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )
    if not model:
        print('指定net不存在，使用默认net')
        model = Classifier()
    # 统计模型参数量以及其他指标
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print("================================================================")
    print(f"Total Parameters: {total_params}")
    print("================================================================")
    return model


if __name__ == '__main__':
    # model = eca_resnet152()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, len(cls))
    # print(model)
    print(Local)
