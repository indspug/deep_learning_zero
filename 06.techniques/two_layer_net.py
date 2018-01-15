# coding: utf-8

import sys, os
sys.path.append(os.getcwd())
#from common.functions import sigmoid, cross_entropy, softmax
from common.gradient import numerical_gradient
from common.layer import Affine, ReLU, SoftmaxWithLoss

import numpy as np
from collections import OrderedDict

##################################################
# 2層のニューラルネットワーク
##################################################
class TwoLayerNet:
	
	# ------------------------------
	# コンストラクタ
	# ------------------------------
	def __init__(self, input_size, hidden_size, output_size, weight_stddev=0.01):
		
		# 正規分布(μ=0, σ=stddev)で重み,バイアスを初期化する
		self.params = {}
		self.params['W1'] = weight_stddev * \
							np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_stddev * \
							np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)
		
		# レイヤの生成
		self.layers = OrderedDict()	# 順番付ディクショナリ
		self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['Func1'] = ReLU()
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
		self.lastLayer = SoftmaxWithLoss()
	
	# ------------------------------
	# 認識
	# ------------------------------
	def predict(self, x):
		y = x
		for layer in self.layers.values():
			y = layer.forward(y)
			#x = layer.forward(x)
		
		return y
		#return x
	
	# ------------------------------
	# 損失関数
	# ------------------------------
	def loss(self, x, t):
		y = self.predict(x)
		loss = self.lastLayer.forward(y, t)
		return loss
	
	# ------------------------------
	# 認識精度
	# ------------------------------
	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		t = np.argmax(t, axis=1)
		n = x.shape[0]
		accuracy = np.sum(y==t) / float(n)
		return accuracy
		
	# ------------------------------
	# 勾配(数値微分)
	# ------------------------------
	def numerical_gradient(self, x, t):
		loss_W = lambda w: self.loss(x, t)
		
		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
		
		return grads
	
	# ------------------------------
	# 勾配(誤差逆伝播法)
	# ------------------------------
	def gradient(self, x, t):
		# forward
		loss = self.loss(x, t)
		
		# backward
		dout = 1
		dout = self.lastLayer.backward(dout)
		
		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)
		
		# 勾配設定
		grads = {}
		grads['W1'] = self.layers['Affine1'].dW 
		grads['b1'] = self.layers['Affine1'].db 
		grads['W2'] = self.layers['Affine2'].dW 
		grads['b2'] = self.layers['Affine2'].db 
		
		return grads
	
