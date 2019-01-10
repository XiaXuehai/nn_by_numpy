# -*- coding: UTF-8 -*-

import cv2
import numpy as np

def pHash(imgfile):
    # 加载并调整图片为32x32灰度图片
    img = cv2.imread(imgfile, 0)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)

    # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img  # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    # 拿到左上角的8 * 8
    vis1 = vis1[0:8, 0:8]

    # 把二维list变成一维list
    img_list = vis1.flatten()


    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i < avg else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 8 * 8, 4)])

hash_value = pHash('img_1.jpg')
print(hash_value)