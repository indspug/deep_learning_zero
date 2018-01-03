# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

##################################################
# ステップ関数
##################################################
def step_function(x):
	
	# (x > 0)でbool型に変換、dtypeでbool型からfloatに変換
	return np.array(x > 0, dtype=np.float)

##################################################
# シグモイド関数
##################################################
def sigmoid_function(x):
	
	return 1 / (1 + np.exp(-x))

##################################################
# ReLU関数
##################################################
def relu_function(x):
	
	return np.maximum(0, x)

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	x = np.arange(-5.0, 5.0, 0.1)
	step_y = step_function(x)
	sigm_y = sigmoid_function(x)
	relu_y = relu_function(x)
	
	plt.plot(x, step_y, label='step', linestyle='-')
	plt.plot(x, sigm_y, label='sigmoid', linestyle='--')
	plt.plot(x, relu_y, label='relu', linestyle='-.')
	plt.ylim(-0.1, 3.1)
	plt.legend()
	plt.show()
	

