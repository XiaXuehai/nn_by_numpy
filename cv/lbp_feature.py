# -*- coding: UTF-8 -*-

import numpy as np
import cv2

def lbp(a):
	b = np.zeros(a.shape, dtype=a.dtype)
	for i in range(1, a.shape[0]-1):
		for j in range(1, a.shape[1]-1):
			code = 0
			x = a[i][j]
			code |= (a[i-1][j-1] >= x) << 7
			code |= (a[i-1][j  ] >= x) << 6
			code |= (a[i-1][j+1] >= x) << 5
			code |= (a[i  ][j+1] >= x) << 4
			code |= (a[i+1][j+1] >= x) << 3
			code |= (a[i+1][j  ] >= x) << 2
			code |= (a[i+1][j-1] >= x) << 1
			code |= (a[i  ][j-1] >= x) << 0
			b[i][j] = code
	return b

src = cv2.imread('../img_1.jpg')
src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
cv2.imshow('src', src)
img_lbp = lbp(src)
cv2.imshow('bb',img_lbp)
cv2.waitKey()