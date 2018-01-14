# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

import numpy as np
import matplotlib.pylab as plt

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# MNISTのデータをロードする
	(x_train, t_train), (x_test, t_test) = \
		load_mnist(flatten=True, normalize=True, one_hot_label=False)
	
	# tをone_hot表現に変換する
	t1_train = np.zeros((t_train.shape[0], 10))
	for i in range(t_train.shape[0]):
		maxidx = t_train[i]
		t1_train[i,maxidx] = 1
	t_train = t1_train
	
	# ハイパーパラメータ
	iter_num = 100					# 繰返し計算回数
	train_size = x_train.shape[0]	# 学習データ数
	batch_size = 50				# バッチサイズ
	learning_rate = 0.1				# 学習係数
	
	# 2層のニューラルネットワーク取得
	network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
	
	# 繰返し計算
	train_loss = []
	for i in range(iter_num):
		
		# ミニバッチの取得
		batch_idx = np.random.choice(train_size, batch_size)
		x_batch = x_train[batch_idx]
		t_batch = t_train[batch_idx]
		
		# 勾配の計算
		grads = network.numerical_gradient(x_batch, t_batch)
		
		# パラメータの更新
		for key in ('W1', 'b1', 'W2', 'b2'):
			network.params[key] -= learning_rate * grads[key]
		
		# 学習経過の記録
		loss = network.loss(x_batch, t_batch)
		train_loss.append(loss)
		
		print('%04d : loss=%f' % (i, loss))
	
	# 学習経過の表示
	epoch = np.arange(len(train_loss))
	plt.plot(epoch, train_loss, label='loss', linestyle='-')
	plt.xlim(0, iter_num)
	plt.ylim(0, max(train_loss))
	plt.xlabel('iteration')
	plt.ylabel('loss')
	plt.show()
	
