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
			conv_dsc(32, 64, 1),
			conv_dsc(64, 128, 2),
			conv_dsc(128, 128, 1),
			conv_dsc(128, 256, 2),
			conv_dsc(256, 256, 1),
			conv_dsc(256, 512, 2),

			conv_dsc(512, 512, 1),
			conv_dsc(512, 512, 1),
			conv_dsc(512, 512, 1),
			conv_dsc(512, 512, 1),
			conv_dsc(512, 512, 1),

			conv_dsc(512, 1024, 2),
			conv_dsc(1024, 1024, 1),
			nn.AvgPool2d(7)
			)
		self.FC = nn.Linear(1024, 1000)
		
	def forward(self, x):
		x = self.model(x)
		x = x.view(-1, 1024)
		x = self.FC(x)
		x = F.softmax(x, dim=1)
		return x


if __name__ == '__main__':
	a = torch.randn(10, 3, 224, 224)
	net = DSC()
	for i in net.named_parameters():
		print('parameters shape:', i[1].shape)
	print('output shape:', net(a).shape)
