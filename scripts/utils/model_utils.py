import torch
import torch.nn as nn

"""
make convolutional layer
"""
def conv(in_planes, out_planes, kernel_size = 3, stride = 1, padding = 1, dilation = 1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size= kernel_size, stride= stride, padding= padding, dilation= dilation, bias= True),
        nn.LeakyReLU(0.1)
    )


"""
make deconvolutional layer
"""
def deconv(in_planes, out_planes, kernel_size = 4, stride = 2, padding= 1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size= kernel_size, stride= stride, padding= padding, bias= True)

