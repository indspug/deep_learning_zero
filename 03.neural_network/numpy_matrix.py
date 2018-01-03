# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 1次元の行列
	print('------------------------------')
	A = np.array([1,2,3,4])
	print('A=', end='')
	print(A)
	print('np.ndim(A)=', end='')
	print(np.ndim(A))
	print('A.shape=', end='')
	print(A.shape)
	
	# 2次元の行列
	print('------------------------------')
	B = np.array([[1,2],[3,4],[5,6]])
	print('B=')
	print(B)
	print('np.ndim(B)=', end='')
	print(np.ndim(B))
	print('B.shape=', end='')
	print(B.shape)
	
	# 行列の内積
	print('------------------------------')
	C = np.array([[1,2],[3,4]])
	D = np.array([[5,6],[7,8]])
	print('np.dot(C,D)=')
	print(np.dot(C,D))

