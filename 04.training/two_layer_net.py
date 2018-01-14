# coding: utf-8

import sys, os
sys.path.append(os.getcwd())
from common.functions import sigmoid, cross_entropy, softmax
from common.gradient import numerical_gradient

import numpy as np

##################################################
# 2層のニューラルネットワーク
##################################################
class TwoLayerNet:
	
	# ------------------------------
	# コンストラクタ
	# ------------------------------
	def __init__(self, input_size, hidden_size, output_size, weight_stddev=0.1):
		
		# 正規分布(μ=0, σ=stddev)で重みを初期化する
		self.params = {}
		self.params['W1'] = weight_stddev * \
							np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_stddev * \
							np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)
	
	# ------------------------------
	# 認識
	# ------------------------------
	def predict(self, x):
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		
		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)
		
		return y
	
	# ------------------------------
	# 損失関数
	# ------------------------------
	def loss(self, x, t):
		y = self.predict(x)
		loss = cross_entropy(y, t)
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
	# 勾配
	# ------------------------------
	def numerical_gradient(self, x, t):
		loss_W = lambda w: self.loss(x, t)
		
		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
		
		return grads

