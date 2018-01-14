# coding: utf-8

import numpy as np
from common.functions import *

##################################################
# ReLU
##################################################
class ReLU:
	
	# コンストラクタ
	def __init__(self):
		self.mask = None
	
	# 順伝播
	def forward(self, x):
		self.mask = (x <= 0)	# 0より大きいならFalse
		fx = x.copy()
		fx[self.mask] = 0		# True(0以下)の要素は不活性(0)にする
		return fx
	
	# 逆伝播
	def backward(self, dout):
		# f'(x) = 1(x>0), 0(x<=0)
		dfx = dout.copy()
		dfx[self.mask] = 0
		return dfx
	
##################################################
# シグモイド関数
##################################################
class Sigmoid:
	
	# コンストラクタ
	def __init__(self):
		self.fx = None
	
	# 順伝播
	def forward(self, x):
		fx = sigmoid(x)
		self.fx = fx
		return fx
	
	# 逆伝播
	def backward(self, dout):
		# f'(x) = (1 - f(x)) * f(x)
		dfx = dout * (1.0 - self.fx) * self.fx
		return dfx
	
##################################################
# アフィン変換
##################################################
class Affine:
	
	# コンストラクタ
	def __init__(self, W, b):
		self.W = W
		self.b = b
		
		self.x = None
		self.original_x_shape = None
		
		self.dW = None
		self.db = None
	
	# 順伝播
	def forward(self, x):
		# 
		self.original_x_shape = x.shape
		x = x.reshape(x.shape[0], -1)
		self.x = x
		
		# u = x * W + b  (N,1,j) * (j,i)
		u = np.dot(self.x, self.W) + self.b
		return u
		
	# 逆伝播
	def backward(self, delta):
		# dL/du = dL/du(l+1) * W.T
		dx = np.dot(delta, self.W.T)
		# dL/dW = dL/du * W.T
		self.dW = np.dot(self.x.T, delta)
		# dL/db = dL/du
		self.db = np.sum(delta, axis=0)
		
		#delta = np.dot(delta, self.W.T)
		dx = dx.reshape(*self.original_x_shape)
		return dx

##################################################
# ソフトマックス(出力層)
##################################################
class SoftmaxWithLoss:
	
	# コンストラクタ
	def __init__(self):
		self.loss = None
		self.y = None	# softmaxの出力
		self.t = None	# 教師データ
	
	# 順伝播
	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy(self.y, self.t)
		return self.loss
		
	# 逆伝播
	def backward(self, dout=1):
		N = self.t.shape[0]
		if self.t.size == self.y.size:
			delta = (self.y - self.t) / N
		else:
			delta = self.y.dopy()
			delta[np.arange(N), self.t] -= 1
			delta = delta / N
		
		return delta

