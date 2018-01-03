# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

##################################################
# 画像の表示
##################################################
def image_show(img):
	
	# byte型に変換してから表示する
	pil_img = Image.fromarray(np.uint8(img))
	pil_img.show()
	

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# MNISTのデータをロードする
	(x_train, t_train), (x_test, t_test) = \
		load_mnist(flatten=True, normalize=False)
	
	# 訓練データの先頭を画像(28x28,白黒)として表示する
	index = 1
	img = x_train[index]
	label = t_train[index]
	img = img.reshape(28,28)
	
	print('label=%d' % label)
	image_show(img)
	
	
	
