# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from multi_layer_net import MultiLayerNet
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
	
	# 過学習を発生させるために学習データを減らす
	random_idx = np.random.choice(x_train.shape[0], 200)
	x_train = x_train[random_idx]
	t_train = t_train[random_idx]
	
	# ハイパーパラメータ
	iter_num = 500			# 繰返し計算回数
	batch_size = 100			# バッチサイズ
	learning_rate = 0.01			# 学習係数
	
	# パラメータ設定
	train_size = x_train.shape[0]	# 学習データ数
	iter_per_epoch = max(train_size / batch_size, 1)
			# 1学習データ学習するための繰返し計算数
	
	# 誤差逆伝播の最適化法取得
	optimizer = Momentum(learning_rate)
	
	# 5層のニューラルネットワーク取得
	networks = {}
	networks['normal'] = MultiLayerNet(input_size=784, \
							hidden_size_list=[50,50,50,50], \
							output_size=10)
	networks['batchnorm'] = MultiLayerNet(input_size=784, \
							hidden_size_list=[50,50,50,50], \
							output_size=10, use_batchnorm=True)
	
	# 損失と精度の履歴リスト初期化
	train_loss_list = {}
	train_acc_list  = {}
	test_acc_list   = {}
	for key in networks.keys():
		train_loss_list[key] = []
		train_acc_list[key]  = []
		test_acc_list[key]   = []
	
	# 繰返し計算
	epoch_num = 0
	for i in range(iter_num):
		
		# ミニバッチの取得
		batch_idx = np.random.choice(train_size, batch_size)
		x_batch = x_train[batch_idx]
		t_batch = t_train[batch_idx]
		
		# 勾配の計算とパラメータ更新
		for key in networks.keys():
			grads = networks[key].gradient(x_batch, t_batch)
			optimizer.update(networks[key].params, grads)
		
		# 1エポック毎に訓練用データの精度と、
		# テスト用データの精度を算出する
		if (i % iter_per_epoch) == 0:
			epoch = i / iter_per_epoch
			epoch_num += 1
			
			for key in networks.keys():
				# 精度を記録する
				train_acc = networks[key].accuracy(x_train, t_train)
				test_acc = networks[key].accuracy(x_test, t_test)
				train_acc_list[key].append(train_acc)
				test_acc_list[key].append(test_acc)
				
				# 損失関数の値を記録する
				loss = networks[key].loss(x_batch, t_batch)
				train_loss_list[key].append(loss)
				print('%04d : [%10s] loss=%.4f train=%.2f test=%.2f' % \
						(epoch, key, loss, train_acc, test_acc) )
	
	fig = plt.figure()
	epoch = np.arange(0, epoch_num)
	
	for i, key in enumerate(train_loss_list.keys()):
		# 損失関数の経過表示
		loss_plot = fig.add_subplot(2, 2, 2*i+1)
		loss_plot.plot(epoch, train_loss_list[key], \
						label='loss', marker='o', markevery=10)
		loss_plot.set_xlim(0, epoch_num)
		loss_plot.set_ylim(0, 3.0)
		loss_plot.set_xlabel('Epoch')
		loss_plot.set_ylabel('Loss')
		
		# 精度の経過表示
		acc_plot = fig.add_subplot(2, 2, 2*i+2)
		acc_plot.plot(epoch, train_acc_list[key], label='Train', marker='s', markevery=10)
		acc_plot.plot(epoch, test_acc_list[key],  label='Test',  marker='x', markevery=10)
		acc_plot.set_xlim(0, epoch_num)
		acc_plot.set_ylim(0, 1.0)
		acc_plot.set_xlabel('Epoch')
		acc_plot.set_ylabel('Acc')
		acc_plot.legend()
	
	plt.show()
	
