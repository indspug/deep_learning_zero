# coding: utf-8

import numpy as np
from common.functions import *

##################################################
# SGD(確率的勾配降下法)
##################################################
class SGD:
	
	# コンストラクタ
	def __init__(self, lr=0.01):
		self.lr = lr
	
	# パラメータ更新
	def update(self, params, grads):
		for key in params.keys():
			params[key] -= self.lr * grads[key]
	

##################################################
# モーメンタム
##################################################
class Momentum:
	
	# コンストラクタ
	def __init__(self, lr=0.01, momentum=0.9):
		self.lr = lr
		self.momentum = momentum
		self.v = None
	
	# パラメータ更新
	def update(self, params, grads):
		if self.v is None:
			self.v = {}
			for key, val in params.items():
				self.v[key] = np.zeros_like(val)
				
		for key in params.keys():
			self.v[key] = (self.momentum * self.v[key]) - (self.lr * grads[key])
			params[key] += self.v[key]
	

