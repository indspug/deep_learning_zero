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
	
	# t_trainをone_hot表現に変換する
	t1_train = np.zeros((t_train.shape[0], 10))
	for i in range(t_train.shape[0]):
		maxidx = t_train[i]
		t1_train[i,maxidx] = 1
	t_train = t1_train
	
	# t_testをone_hot表現に変換する
	t1_test = np.zeros((t_test.shape[0], 10))
	for i in range(t_test.shape[0]):
		maxidx = t_test[i]
		t1_test[i,maxidx] = 1
	t_test = t1_test
	
	# ハイパーパラメータ
	iter_num = 100000			# 繰返し計算回数
	batch_size = 100			# バッチサイズ
	learning_rate = 0.1			# 学習係数
	
	# パラメータ設定
	train_size = x_train.shape[0]	# 学習データ数
	iter_per_epoch = max(train_size / batch_size, 1)
			# 1学習データ学習するための繰返し計算数
	
	# 2層のニューラルネットワーク取得
	network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
	
	# 繰返し計算
	train_loss = []
	train_acc_history = []
	test_acc_history = []
	for i in range(iter_num):
		
		# ミニバッチの取得
		batch_idx = np.random.choice(train_size, batch_size)
		x_batch = x_train[batch_idx]
		t_batch = t_train[batch_idx]
		
		# 勾配の計算
		grads = network.gradient(x_batch, t_batch)
		
		# パラメータの更新
		for key in ('W1', 'b1', 'W2', 'b2'):
			network.params[key] -= learning_rate * grads[key]
		
		# 学習経過の記録
		loss = network.loss(x_batch, t_batch)
		train_loss.append(loss)
		
		# 1エポック毎に精度を記録する
		if (i % iter_per_epoch) == 0:
			train_acc = network.accuracy(x_train, t_train)
			test_acc = network.accuracy(x_test, t_test)
			train_acc_history.append(train_acc)
			test_acc_history.append(test_acc)
			print('%04d : loss=%f' % (i/iter_per_epoch, loss))
	
	fig = plt.figure()
	
	# 損失関数の経過表示
	loss_plot = fig.add_subplot(2, 1, 1)
	iteration = np.arange(len(train_loss))
	loss_plot.plot(iteration, train_loss, label='loss', linestyle='-')
	loss_plot.set_xlim(0, iter_num)
	loss_plot.set_ylim(0, max(train_loss))
	loss_plot.set_xlabel('Iteration')
	loss_plot.set_ylabel('Loss')
	
	# 精度の経過表示
	acc_plot = fig.add_subplot(2, 1, 2)
	epoch_num = len(train_acc_history)
	epoch = np.arange(epoch_num)
	acc_plot.plot(epoch, train_acc_history, label='Train Acc', color='blue', linestyle='--')
	acc_plot.plot(epoch, test_acc_history, label='Test Acc', color='red', linestyle='-')
	acc_plot.set_xlim(0, epoch_num)
	acc_plot.set_ylim(0, 1.0)
	acc_plot.set_xlabel('Epoch')
	acc_plot.set_ylabel('Accuracy')
	acc_plot.legend(loc=1)
	
	plt.show()
	
