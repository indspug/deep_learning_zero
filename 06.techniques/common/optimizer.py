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
	

##################################################
# AdamGrad
##################################################
class AdamGrad:
	
	# コンストラクタ
	def __init__(self, lr=0.01):
		self.lr = lr
		self.h = None
	
	# パラメータ更新
	def update(self, params, grads):
		if self.h is None:
			self.h = {}
			for key,val in params.items():
				self.h[key] = np.zeros_like(val)
		
		for key in params.keys():
			self.h[key] += grads[key] * grads[key]
			params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
		

##################################################
# Adam
##################################################
class Adam:
	
	"""Adam (http://arxiv.org/abs/1412.6980v8)"""
	
	# コンストラクタ
	def __init__(self, lr=0.001, beta1=0.9, beta2=0.99):
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.iter = 0
		self.m = None
		self.v = None
	
	# パラメータ更新
	def update(self, params, grads):
		if self.m is None:
			self.m, self.v = {}, {}
			for key, val in params.items():
				self.m[key] = np.zeros_like(val)
				self.v[key] = np.zeros_like(val)
				
		self.iter += 1
		lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / \
								(1.0 - self.beta1 ** self.iter)
		
		for key in params.keys():
			self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
			self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
			params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
			
	

