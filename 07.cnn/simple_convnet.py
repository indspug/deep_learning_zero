# -*- coding: utf-8 -*-
  
import sys, os
sys.path.append(os.getcwd())
from common.layers import *
from common.gradient import numerical_gradient
import numpy as np
from collections import OrderedDict

##################################################
# シンプルなCNN
##################################################
class SimpleConvNet:
	
	# Conv - ReLU - Pool - Affine - ReLU - Affine - SoftMax
	
	"""
	input_dim: 入力サイズ(MNISTの場合は(1,28,28))
	hidden_size: 隠れ層のニューロンの数(e.g. 100)
	output_size: 出力サイズ(MNISTの場合は10)
	activation: 'relu'
	"""
	# ------------------------------
	# コンストラクタ
	# ------------------------------
	def __init__(self, input_dim=(1, 28, 28), 
					conv_param={'filter_num':30, 'filter_size':5, 'padding':0, 'stride':1},
					pool_param={'size':2, 'padding':0, 'stride':2},
					hidden_size=100, output_size=10):
		filter_num   = conv_param['filter_num']
		filter_size  = conv_param['filter_size']
		conv_padding = conv_param['padding']
		conv_stride  = conv_param['stride']
		pool_size    = pool_param['size']
		pool_padding = pool_param['padding']
		pool_stride  = pool_param['stride']
		input_c = input_dim[0]
		input_h = input_dim[1]
		input_w = input_dim[2]
		conv_out_h = int( (input_h + 2*conv_padding - filter_size) / conv_stride) + 1
		conv_out_w = int( (input_w + 2*conv_padding - filter_size) / conv_stride) + 1
		pool_out_h = int( (conv_out_h - pool_size) / pool_stride ) + 1
		pool_out_w = int( (conv_out_h - pool_size) / pool_stride ) + 1
		self.params = {}
		self.layers = OrderedDict()
		
		# 第1層の重み初期化
		layer1_size = filter_num * input_c * filter_size * filter_size
		scale = np.sqrt(2.0 / layer1_size)
		self.params['Conv1_W'] = \
			scale * np.random.randn(filter_num, input_c, filter_size, filter_size)
		self.params['Conv1_b'] = np.zeros(filter_num)
		
		# 第2層の重み初期化
		layer2_size = (filter_num * pool_out_h * pool_out_w) * hidden_size
		scale = np.sqrt(2.0 / layer2_size)
		self.params['Affine1_W'] = \
			scale * np.random.randn(filter_num * pool_out_h * pool_out_w, hidden_size)
		self.params['Affine1_b'] = np.zeros(hidden_size)
		
		# 第3層の重み初期化
		layer3_size = hidden_size * output_size
		scale = np.sqrt(2.0 / layer3_size)
		self.params['Affine2_W'] = scale * np.random.randn(hidden_size, output_size)
		self.params['Affine2_b'] = np.zeros(output_size)
		
		# レイヤの作成
		self.layers['Conv1'] = Convolution(self.params['Conv1_W'], self.params['Conv1_b'],
											conv_stride, conv_padding)
		self.layers['ReLU1'] = ReLU()
		self.layers['Pool1'] = Pooling(pool_size, pool_size, pool_stride, pool_padding)
		self.layers['Affine1'] = Affine(self.params['Affine1_W'], self.params['Affine1_b'])
		self.layers['ReLU2'] = ReLU()
		self.layers['Affine2'] = Affine(self.params['Affine2_W'], self.params['Affine2_b'])
		self.lastLayer = SoftmaxWithLoss()
	
	# ------------------------------
	# 認識
	# ------------------------------
	def predict(self, x):
		y = x
		for key, layer in self.layers.items():
			y = layer.forward(y)
		
		return y
	
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
	# 勾配(誤差逆伝播法)
	# ------------------------------
	def gradient(self, x, t):
		
		# 順伝播計算
		loss = self.loss(x, t)
		
		# 逆伝播計算
		dout = 1
		dout = self.lastLayer.backward(dout)
		
		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)
		
		# 勾配設定
		grads = {}
		grads['Conv1_W'] = self.layers['Conv1'].dW
		grads['Conv1_b'] = self.layers['Conv1'].db
		grads['Affine1_W'] = self.layers['Affine1'].dW
		grads['Affine1_b'] = self.layers['Affine1'].db
		grads['Affine2_W'] = self.layers['Affine2'].dW
		grads['Affine2_b'] = self.layers['Affine2'].db
		
		return grads
	
