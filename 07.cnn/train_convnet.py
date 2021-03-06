# -*- coding: utf-8 -*-
  
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

sys.path.append(os.getcwd())
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

import numpy as np
import matplotlib.pyplot as plt

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# MNISTのデータをロードする
	(x_train, t_train), (x_test, t_test) = \
		load_mnist(flatten=False, normalize=True, one_hot_label=True)
	
	#print(x_train.shape)
	
	max_epochs = 20
	
	network = SimpleConvNet(
				input_dim=(1,28,28), 
				conv_param = {'filter_num': 20, 'filter_size': 5, 
								'padding': 0, 'stride': 1},
				hidden_size=30, output_size=10)
	                        
	trainer = Trainer(network, x_train, t_train, x_test, t_test,
						epochs=max_epochs, mini_batch_size=100,
						optimizer='Adam', optimizer_param={'lr': 0.001},
						evaluate_sample_num_per_epoch=1000)
	
	trainer.train()
	
	# パラメータの保存
	#network.save_params("params.pkl")
	#print("Saved Network Parameters!")
	
	# グラフの描画
	markers = {'train': 'o', 'test': 's'}
	x = np.arange(max_epochs)
	plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
	plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.ylim(0, 1.0)
	plt.legend(loc='lower right')
	plt.show()
	
