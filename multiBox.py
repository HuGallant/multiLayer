import torch
import torch.nn as nn

def vgg(cfg, i, batch_norm=False):
    
    layers = []
    in_channels = i
    for v in cfg:
        # 正常的 max_pooling
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # ceil_mode = True, 上采样使得 channel 75-->38
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            # update in_channels
            in_channels = v

    # max_pooling (3,3,1,1)        
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # 新添加的网络层 1024x3x3
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

    # 新添加的网络层 1024x1x1
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    # 结合到整体网络中
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


"""
# 代码测试
if __name__ == "__main__":
    base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
    }
    vgg = nn.Sequential(*vgg(base['300'], 3))
    x = torch.randn(1,3,300,300)
    print(vgg(x).shape)  #(1, 1024, 19, 19)
"""
def add_extras(cfg, i, batch_norm=False):
    '''
    为后续多尺度提取，增加网络层
    '''
    layers = []
    # 初始输入通道为 1024
    in_channels = i
    # flag 用来选择 kernel_size= 1 or 3
    flag = False
    for k,v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k+1], 
                                    kernel_size=(1,3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]

            flag = not flag # 反转flag

        in_channels = v # 更新 in_channels

    return layers

def multibox(vgg, extra_layers, cfg, num_classes):
    '''
    Args:
        vgg: 修改fc后的vgg网络
        extra_layers: 加在vgg后面的4层网络
        cfg: 网络参数，eg:[4, 6, 6, 6, 4, 4]
        num_classes: 类别，VOC为 20+背景=21
    Return:
        vgg, extra_layers
        loc_layers: 多尺度分支的回归网络
        conf_layers: 多尺度分支的分类网络
    '''
    loc_layers = []
    conf_layers = []
    vgg_layer = [21, -2]
    # 第一部分，vgg 网络的 Conv2d-4_3(21层)， Conv2d-7_1(-2层)
    for k, v in enumerate(vgg_layer):
        # 回归 box*4(坐标)
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k]*4, kernel_size=3, padding=1)]     
        # 置信度 box*(num_classes)
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k]*num_classes, kernel_size=3, padding=1)]    

    # 第二部分，cfg从第三个开始作为box的个数，而且用于多尺度提取的网络分别为1,3,5,7层
    for k, v in enumerate(extra_layers[1::2],2):
        # 回归 box*4(坐标)
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]*4, kernel_size=3, padding=1)]
        # 置信度 box*(num_classes)
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]*(num_classes), kernel_size=3, padding=1)]

    return vgg, extra_layers, (loc_layers, conf_layers)
if __name__  == "__main__":
    base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
    }
    
    extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
    }
    vgg, extra_layers, (l, c) = multibox(vgg(base['300'], 3),
                                         add_extras(extras['300'], 1024),
                                         [4, 6, 6, 6, 4, 4], 21)
    print(nn.Sequential(*l))
    print('---------------------------')
    print(nn.Sequential(*c))
    #upload to github