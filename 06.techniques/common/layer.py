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
		
		# u = x * W + b		(N,1,j) * (j,i) -> (N,i)
		#					ex. (100,1,784) * (784,50) -> (100, 50)
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

##################################################
# BatchNormalization
##################################################
class BatchNormalization:
	
	# コンストラクタ
	def __init__(self, gamma, beta, momentum=0.9, \
					running_mean=None, running_var=None):
		self.gamma = gamma
		self.beta = beta
		self.momentum = momentum
		self.input_shape = None
		
		# テスト時に使用する平均と分散
		self.running_mean = running_mean
		self.running_var  = running_var
		
		# 逆伝播(backward)時に使用する中間データ
		self.batch_size = None
		self.xc = None
		self.std = None
		self.dgamma = None
		self.dbeta = None
		
	# 順伝播
	def forward(self, x, train_flg=True):
		self.input_shape = x.shape
		
		# xを(N,)形式に変換(畳み込み計算用)
		if x.ndim != 2:
			N, C, H, W = x.shape
			x = x.reshape(N, -1)	# (N, C*H*W)
		
		out = self.__forward(x, train_flg)
		
		# 形式を入力と同じに戻してReturn
		return out.reshape(*self.input_shape)
		
	# 順伝播計算
	def __forward(self, x, train_flg):
		
		if self.running_mean is None:
			N, D = x.shape
			self.running_mean = np.zeros(D)
			self.running_var  = np.zeros(D)
		
		if train_flg:
			# 学習時
			mu = x.mean(axis=0)		# μ = 1/N * Σx
			xc = x - mu				# xc = x - μ
			var = np.mean(xc**2, axis=0)	# σ^2 = Σ(x-μ)^2
			std = np.sqrt(var + 10e-7)		# σ^2 - ε
			xn = xc / std	# x^ = (x - μ) / √(σ^2 - ε)
			
			self.batch_size = x.shape[0]
			self.xc = xc
			self.xn = xn
			self.std = std
			self.running_mean = self.momentum * self.running_mean + \
								(1 - self.momentum) * mu
			self.running_var  = self.momentum * self.running_var + \
								(1 - self.momentum) * var
		else:
			# テスト時
			xc = x - self.running_mean
			xn = xc / (np.sqrt(self.running_var + 10e-7))
		
		out = self.gamma * xn + self.beta
		return out
	
	# 逆伝播
	def backward(self, dout=1):
		
		# δを(N,)形式に変換(畳み込み計算用)
		if dout.ndim != 2:
			N,C,H,W = dout.shape
			dout = dout.reshape(N,-1)	# (N,C*H*W)
		
		# 元の形式に戻してReturn
		delta = self.__backward(dout)
		delta = delta.reshape(*self.input_shape)
		
		return delta
		
	# 逆伝播計算
	def __backward(self, dout):
		# y = (γ * x^) + β
		dbeta = dout.sum(axis=0)	# dβ = Σdout
		dgamma = np.sum(self.xn * dout, axis=0)	# dγ = Σ(dout * x^)
		dxn = self.gamma * dout		# dx^ = γ * dout
		
		# x^ = (x - μ) / √(σ^2 - ε)
		dxc = dxn / self.std
			# d(x-μ) = dx^ / √(σ^2 - ε)
		
		# x^ = (x - μ) * t 
		#		(t = 1 / √(σ^2 - ε))
		# t = 1 / u
		#		(u = √(σ^2 - ε))
		# u = √v
		#		(v = (σ^2 - ε))
		dt = np.sum(dxn * self.xc, axis=0)
			# dt = (x-μ) * Σdx^
		du = - dt / (self.std * self.std)
			# du = dt * (- 1 / u^2)
		dvar = du * 0.5 / self.std
			# dv = du * 0.5 / √v
		#dvar = dstd * 0.5 / self.std
		#dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
				# dσ = d√(σ^2 - ε) * (0.5 / √(σ^2 - ε))
		
		# σ^2 = 1/N * Σ(x-μ)^2
		dxc += dvar * (2.0 / self.batch_size) * self.xc
				# d(x-μ) = dx^ / √(σ^2 - ε) + 
				#          dvar * 1/N * 2 * (x-μ)
		# x-μ
		dx1 = dxc					# dx1 = dxc * 1
		dmu = -np.sum(dxc, axis=0)	# dμ = Σdxc * (-1)
		
		# μ = 1/N * Σx
		dx2 = dmu / self.batch_size	# dμ = dμ * 1/N * 1
		
		dx = dx1 + dx2
		
		# dγ, dβをクラス変数に設定
		self.dgamma = dgamma
		self.dbeta  = dbeta
		
		return dx
		

