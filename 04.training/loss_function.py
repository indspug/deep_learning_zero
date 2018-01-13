# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np

DELTA = 1e-7

##################################################
# データ(MNIST)取得
##################################################
def get_data():
	
	# MNISTのデータをロードする
	(x_train, t_train), (x_test, t_test) = \
		load_mnist(flatten=True, normalize=True, one_hot_label=False)
	
	train_data = {'X':x_train, 'T':t_train}
	test_data  = {'X':x_test,  'T':t_test }
	
	return (train_data, test_data)

##################################################
# 二乗和誤差
##################################################
def mean_squared_error(y, t):
	
	print('y.shape=', end='')
	print(y.shape)
	print('y.size=%d'% y.size)
	print('y.ndim=%d'% y.ndim)
	
	if y.ndim == 1:
		y = y.reshape(1, y.size)
		t = t.reshape(1, t.size)
	
	n = y.shape[0]
	loss = 0.5 * np.sum((y-t)**2)
	return (loss / n)

##################################################
# クロスエントロピー
##################################################
def cross_entropy(y, t):
	
	if y.ndim == 1:
		y = y.reshape(1, y.size)
		t = t.reshape(1, t.size)
	
	n = y.shape[0]
	# 正解データがone-host表現ではなく、
	# [3],[7]といったラベルで与えられるとき、
	# yから正解ラベルと対応するインデックスのみ取得する
	#   np.arrange(batch_size) = [0,1,2,...] : 0から順にデータ参照するだけ
	#   t : 正解ラベルに対応するインデックス取得
	y1 = y[np.arange(batch_size), t]
	loss = -np.sum(np.log(y1 + DELTA))
	#loss = -np.sum(t * np.log(y))
	return (loss / n)

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# MNISTデータと学習済みパラメータ取得
	train_data, test_data = get_data()
	#network = init_network()
	X = train_data['X']
	T = train_data['T']
	Y = np.zeros((T.shape[0],10))
	
	batch_size = 10
	batch_mask = np.random.choice(T.shape[0], batch_size)
	Y1 = Y[batch_mask]
	T1 = T[batch_mask]
	
	print('batch_mask=%s' % batch_mask)
	
	mse = mean_squared_error(Y1, T1)
	print('Mean Squared Error=%f' % mse)
	
	ce = cross_entropy(Y1, T1)
	print('Cross Entropy=%f' % ce)
