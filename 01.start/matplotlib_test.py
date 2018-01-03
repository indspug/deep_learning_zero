import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	# データ作成
	x = np.arange(0, 6, 0.1)
	y = np.sin(x)
	
	# グラフの描画
	plt.plot(x, y)
	plt.show()
