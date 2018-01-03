# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 1層のニューラルネットワーク
	X = np.array([1, 2])
	W = np.array([ [1,3,5], [2,4,6] ])
	Y = np.dot(X, W)
	
	print('%s*%s->%s' % (X.shape, W.shape, Y.shape))
	print('Y=%s' % Y)

