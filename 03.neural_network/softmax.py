# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

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
# メイン
##################################################
if __name__ == '__main__':
	
	x = [990, 1000, 1010]
	y = softmax_function(x)
	log_y = np.log(y)
	#v100 = np.emtpy(y.shape[0]).fill(100)
	v100 = np.empty(3)
	v100.fill(100)
	
	print('x=%s' % x)
	print('y=%s' % y)
	print('v100=%s' % v100)
	print('y*100=%s' % (y*v100))
	print('log(y)=%s' % log_y)
	
	x1 = np.arange(990, 1010, 1)
	y1 = softmax_function(x1)
	plt.plot(x1, y1)
	plt.ylim(-0.1,1.1)
	plt.show()

