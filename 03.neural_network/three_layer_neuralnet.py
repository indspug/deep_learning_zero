# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

##################################################
# シグモイド関数
##################################################
def sigmoid_function(x):
	return 1 / (1 + np.exp(-x))

##################################################
# 恒等写像
##################################################
def identity_function(x):
	return np.array(x)

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 3層のニューラルネットワーク
	X = np.array([1.0, 0.5])
	
	# 1層目の計算
	W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
	B1 = np.array([0.1, 0.2, 0.3])
	A1 = np.dot(X, W1) + B1
	Z1 = sigmoid_function(A1)
	
	# 2層目の計算
	W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
	B2 = np.array([0.1, 0.2])
	A2 = np.dot(Z1, W2) + B2
	Z2 = sigmoid_function(A2)
	
	# 3層目の計算
	W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
	B3 = np.array([0.1, 0.2])
	A3 = np.dot(Z2, W3) + B3
	Y = identity_function(A3)
	
	print('X:%s -> Z1:%s -> Z2:%s -> Y:%s' % 
			(X.shape, Z1.shape, Z2.shape, Y.shape))
	print('A1=%s' % A1)
	print('Z1=%s' % Z1)
	print('A2=%s' % A2)
	print('Z2=%s' % Z2)
	print('A3=%s' % A3)
	print('Y =%s' % Y)
	
	
	

