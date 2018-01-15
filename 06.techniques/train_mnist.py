# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from common.optimizer import *

import numpy as np
import matplotlib.pylab as plt

##################################################
# 教師データをone_hot表現に変換する
##################################################
def label_to_one_hot(t, label_num):
	
	t1 = np.zeros((t.shape[0], label_num))
	for i in range(t.shape[0]):
		maxidx = t[i]
		t1[i,maxidx] = 1
	
	return t1
	
##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# MNISTのデータをロードする
	(x_train, t_train), (x_test, t_test) = \
		load_mnist(flatten=True, normalize=True, one_hot_label=False)
	
	# 教師データをone_hot表現に変換する
	t_train = label_to_one_hot(t_train, 10)
	t_test = label_to_one_hot(t_test, 10)
	
	# ハイパーパラメータ
	iter_num = 10000			# 繰返し計算回数
	batch_size = 100			# バッチサイズ
	learning_rate = 0.1			# 学習係数
	
	# パラメータ設定
	train_size = x_train.shape[0]	# 学習データ数
	iter_per_epoch = max(train_size / batch_size, 1)
			# 1学習データ学習するための繰返し計算数
	
	# 誤差逆伝播の最適化法取得
	optimizers = {}
	optimizers['SGD'] = SGD()
	optimizers['Momentum'] = Momentum()
	
	network = {}
	train_loss = {}
	for key in optimizers.keys():
		# 2層のニューラルネットワーク取得
		network[key] = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
		train_loss[key] = []
	
	# 繰返し計算
	for i in range(iter_num):
		
		# ミニバッチの取得
		batch_idx = np.random.choice(train_size, batch_size)
		x_batch = x_train[batch_idx]
		t_batch = t_train[batch_idx]
		
		# 全optimizerで計算
		for key in optimizers.keys():
			# 勾配の計算
			grads = network[key].gradient(x_batch, t_batch)
			
			# パラメータ更新
			optimizers[key].update(network[key].params, grads)
			
			# 学習経過の記録
			loss = network[key].loss(x_batch, t_batch)
			train_loss[key].append(loss)
		
		# 1エポック毎に損失関数の数値を表示する
		if (i % iter_per_epoch) == 0:
			epoch = i / iter_per_epoch
			print('epoch : %04d' % epoch)
			for key in optimizers.keys():
				loss = network[key].loss(x_batch, t_batch)
				print('%10s : loss=%f' % (key, loss))
	
	fig = plt.figure()
	
	# 損失関数の経過表示
	loss_plot = fig.add_subplot(1, 1, 1)
	markers = {'SGD':'o', 'Momentum':'x'}
	iteration = np.arange(iter_num)
	for key in optimizers.keys():
		loss_plot.plot(iteration, train_loss[key], label='loss', marker=markers[key])
		
	loss_plot.set_xlim(0, iter_num)
	loss_plot.set_ylim(0, 1.0)
	loss_plot.set_xlabel('Iteration')
	loss_plot.set_ylabel('Loss')
	
	plt.show()
	
