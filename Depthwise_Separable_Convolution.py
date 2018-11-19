# coding: utf-8
# depthwise separable convolution, a part of mobilenet 

import torch
import torch.nn as nn
from torch.nn import functional as F


class DSC(nn.Module):
	def __init__(self):
		super(DSC, self).__init__()
		def conv_bn(inp, oup, stride):
			return nn.Sequential(
				nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
				nn.BatchNorm2d(oup),
				nn.ReLU(inplace=True)
				)
		def conv_dsc(inp, oup, stride):
			return nn.Sequential(
				nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
				nn.BatchNorm2d(inp),
				nn.ReLU(inplace=True),

				nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
				nn.ReLU(inplace=True)
				)

		self.model = nn.Sequential(
			conv_bn(3, 32, 2),
			conv_dsc(32, 64, 1)
			)
		
	def forward(self, x):
		x = self.model(x)
		return x


if __name__ == '__main__':
	a = torch.randn(1, 3, 5, 5)
	net = DSC()
	for i in net.named_parameters():
		print('parameters shape:', i[1].shape)
	print('output shape:', net(a).shape)
