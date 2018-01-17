# -*- coding: utf-8 -*-

#import sys, os
#sys.path.append(os.pardir)
from common.functions import *
import numpy as np
import matplotlib.pyplot as plt

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	node_num = 100			# 各隠れ層のノードの数
	input_data = np.random.rand(1000, node_num)	# 入力データ1000個
	hidden_layer_size = 5	# 隠れ層の数
	activations = {}		# 隠れ層の活性化関数の結果
	
	x = input_data
	for i in range(hidden_layer_size):
		
		# 次の隠れ層の入力セット
		if (i != 0):
			x = activations[i-1]
		
		# 隠れ層の初期値設定
		#w = np.random.randn(node_num, node_num) * 1
		#w = np.random.randn(node_num, node_num) * 0.01
		#w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
		w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
		
		# 隠れ層の計算
		a = np.dot(x, w)
		
		# 活性化関数にかける
		#z = sigmoid(a)
		z = relu(a)
		
		# 活性化関数の結果
		activations[i] = z
		
	# ヒストグラムを描画
	for i, active in activations.items():
		plt.subplot(1, hidden_layer_size, i+1)
		plt.title(str(i+1) + "-layer")
		if (i != 0):
			plt.yticks([], [])
		plt.hist(active.flatten(), 20, range=(0,1))
	plt.show()

