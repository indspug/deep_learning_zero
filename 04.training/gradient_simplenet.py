# coding: utf-8

import numpy as np

DELTA = 1e-7
H = 1e-4

##################################################
# ソフトマックス関数
##################################################
def softmax(x):
	
	# そのまま計算するとオーバーフローするので
	# 最大値を引いた値でexpの計算を行う
	c = np.max(x)
	exp_a = np.exp(x - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	
	return y

##################################################
# クロスエントロピー
##################################################
def cross_entropy(y, t):
	
	# yが1次元の場合は、2次元データに変換
	if y.ndim == 1:
		y = y.reshape(1, y.size)
		t = t.reshape(1, t.size)
	
	# 教師データがラベルの場合、
	# one-hot-vectorに変換
	if t.size != y.size:
		t1 = np.zeros_like(y)
		for i in range(t.shape[0]):
			maxidx = t[i].argmax()
			t1[i,maxidx] = 1
		t = t1
	
	n = y.shape[0]
	loss = -np.sum(t * np.log(y+DELTA))
	
	return (loss / n)

##################################################
# 1層のニューラルネットワーク
##################################################
class simpleNet:
	
	# コンストラクタ
	def __init__(self):
		self.W = np.random.randn(2,3)
	
	# 予測
	def predict(self, x):
		return np.dot(x, self.W)
	
	# 損失関数
	def loss(self, x, t):
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy(y, t)
	
		return loss
	
	# 勾配
	def gradient_LW(self, x, t):
		
		grad = np.zeros_like(self.W)
		W_bk = np.copy(self.W)
		
		it = np.nditer(self.W, flags=['multi_index'], op_flags=['readwrite'])
		while not (it.finished):
			
			# インデックス取得
			idx = it.multi_index
			
			# f(x-h)の計算
			self.W[idx] = W_bk[idx] - H
			L1 = self.loss(x, t)
			
			# f(x+h)の計算
			self.W[idx] = W_bk[idx] + H
			L2 = self.loss(x, t)
			
			# 勾配計算
			grad[idx] = (L2 - L1) / (2*H)
			
			# Wを元に戻す
			self.W = W_bk
			
			# イテレータを次に進める
			it.iternext()
			
		return grad
		
		#for i in range(self.W.shape[0]):
		#	for j in range(self.W.shape[1]):
		#		
		#		# f(x-h)の計算
		#		self.W[i,j] = W_bk[i,j] - H
		#		L1 = self.loss(x, t)
		#		
		#		# f(x+h)の計算
		#		self.W[i,j] = W_bk[i,j] + H
		#		L2 = self.loss(x, t)
		#		
		#		# 勾配計算
		#		grad[i] = (L2 - L1) / (2*H)
		#		
		#		# Wを元に戻す
		#		self.W = W_bk
		#
		#return grad

##################################################
# 重みWの勾配計算用関数
##################################################
def L(w):
	
	net.W = w
	
	return net.loss

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	net = simpleNet()
	print(net.W)
	
	x = np.array([0.6, 0.9])
	p = net.predict(x)
	print(p)
	
	t = np.array([0,0,1])
	print(net.loss(x,t))
	
	dW = net.gradient_LW(x, t)
	print(dW)
	
	
