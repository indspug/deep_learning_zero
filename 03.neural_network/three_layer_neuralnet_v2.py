# -*- coding: utf-8 -*-
  
import numpy as np

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
# ネットワークの初期化
##################################################
def init_network():
	
	network = {}
	network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
	network['B1'] = np.array([0.1, 0.2, 0.3])
	network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
	network['B2'] = np.array([0.1, 0.2])
	network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
	network['B3'] = np.array([0.1, 0.2])
	
	return network

##################################################
# 順伝播計算
##################################################
def forward(network, x):
	
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	B1, B2, B3 = network['B1'], network['B2'], network['B3']
	A1 = np.dot(X, W1) + B1
	Z1 = sigmoid_function(A1)
	A2 = np.dot(Z1, W2) + B2
	Z2 = sigmoid_function(A2)
	A3 = np.dot(Z2, W3) + B3
	Y = identity_function(A3)

	return Y

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 3層のニューラルネットワーク
	network = init_network()
	X = np.array([1.0, 0.5])
	Y = forward(network, X)
	print(Y)
	
