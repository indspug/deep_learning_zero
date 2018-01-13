# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches

H = 1e-4

##################################################
# x0**2 + x1**2
##################################################
def f1(x):
	
	f = x[0]**2 + x[1]**2
	return f

##################################################
# 勾配
##################################################
def gradient(f, x):
	
	grad = np.zeros_like(x)
	xc = np.copy(x)
	
	for i in range(x.size):
		
		# f(x-h)の計算
		xc[i] = x[i] - H
		f1 = f(xc)
		
		# f(x+h)の計算
		xc[i] = x[i] + H
		f2 = f(xc)
		
		# 勾配計算
		grad[i] = (f2 - f1) / (2*H)
		
		# xを元に戻す
		xc[i] = x[i]
	
	return grad

##################################################
# 勾配降下法
##################################################
def gradient_descent(f, init_x, lr=0.01, step_num=100):
	# lr=learning_rage:学習率
	
	x = init_x
	x_history = []
	
	for i in range(step_num):
		x_history.append(x.copy())
		
		grad = gradient(f, x)
		x = x - (lr * grad)
		
	
	return x, np.array(x_history)

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	init_x = np.array([-3.0, 4.0])
	
	x, x_history = gradient_descent(f1, init_x, 0.1, 100)
	
	print('init_x=%s' % init_x)
	print('end_x =%s' % x)
	
	# xの履歴をプロット
	plt.plot(x_history[:,0],x_history[:,1], 'o')
	
	# x_history[0] :	(x0,x1)の初期値 
	#						ex) [-3.0, 4.0]
	# x_history[:,0] : 	x0の履歴
	#						ex) [-3., -2.4, -1.92, ...  0.00]
	
	# 等高線作成
	#plt.plot( [-5, 5], [0, 0], '--b')
	#plt.plot( [0, 0], [-5, 5], '--b')
	ax = plt.axes()
	for i, r in enumerate(np.arange(0.5, 5.0, 0.5)):
		
		c = patches.Circle(xy=(0,0), radius=r, 
							facecolor='None', edgecolor='b', 
							linestyle='--')
		ax.add_patch(c)
	
	# 表示範囲・ラベル設定
	plt.xlim(-4.5, 4.5)
	plt.ylim(-4.5, 4.5)
	plt.xlabel('x0')
	plt.ylabel('x1')
	
	# グラフ表示
	plt.show()
	
