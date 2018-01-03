# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import pickle
import time

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
# ソフトマックス関数
##################################################
def softmax_function(x):
	
	# そのまま計算するとオーバーフローするので
	# 最大値を引いた値でexpの計算を行う
	c = np.max(x)
	exp_a = np.exp(x - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	
	return y

##################################################
# データ(MNIST)取得
##################################################
def get_data():
	
	# MNISTのデータをロードする
	(x_train, t_train), (x_test, t_test) = \
		load_mnist(flatten=True, normalize=True, one_hot_label=False)
	
	train_data = {'X':x_train, 'T':t_train}
	test_data = {'X':x_test, 'T':t_test}
	
	return (train_data, test_data)

##################################################
# ネットワークの初期化
##################################################
def init_network():
	
	# sample_weight.pklから重みとバイアスを取得
	with open('sample_weight.pkl', 'rb') as f:
		network = pickle.load(f)
	
	return network

##################################################
# 予測(順伝播計算)を行う
##################################################
def predict(network, X):
	
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	B1, B2, B3 = network['b1'], network['b2'], network['b3']
	A1 = np.dot(X, W1) + B1
	Z1 = sigmoid_function(A1)
	A2 = np.dot(Z1, W2) + B2
	Z2 = sigmoid_function(A2)
	A3 = np.dot(Z2, W3) + B3
	Y = softmax_function(A3)
	
	return(Y)

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# MNISTデータと学習済みパラメータ取得
	train_data, test_data = get_data()
	network = init_network()
	X = test_data['X']
	T = test_data['T']
	
	# MNISTデータの予測実行
	# 正解と比較して予測の精度を算出する
	st1 = time.time()
	accuracy_cnt = 0
	for i in range(len(X)):
		Y = predict(network, X[i])
		P = np.argmax(Y)
		if (P == T[i]):
			accuracy_cnt += 1
		
	accuracy = float(accuracy_cnt / len(X))
	ed1 = time.time()
	print('==============================')
	print(' 1データずつ計算した場合')
	print('   Accuracy=%.3f' % accuracy)
	print('   Elapsed time[sec]=%.2f' % (ed1 - st1))
	
	# バッチ計算でMNISTデータの予測実行
	st2 = time.time()
	batch_size = 100
	accuracy_cnt = 0
	for i in range(0, len(X), batch_size):
		x_batch = X[i:i+batch_size]
		y_batch = predict(network, x_batch)
		p = np.argmax(y_batch, axis=1)
		accuracy_cnt += np.sum(p == T[i:i+batch_size])
	
	accuracy = float(accuracy_cnt / len(X))
	ed2 = time.time()
	print('==============================')
	print(' バッチ計算した場合')
	print('   Accuracy=%.3f' % accuracy)
	print('   Elapsed time[sec]=%.2f' % (ed2 - st2))
	
