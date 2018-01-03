import numpy as np

##################################################
# ANDゲート
##################################################
def AND(x1, x2):
	
	#x = np.array([[x1, x2]]).T;
	x = np.array([[x1], [x2]]);
	w = np.array([[0.5, 0.5]]);
	b = -0.7
	
	y = np.dot(w, x) + b;
	#x = x.T
	#y = np.sum(w * x) + b;
	
	#print (w)
	#print (x)
	#print (y)
	if y > 0:
		return 1
	else:
		return 0
	

##################################################
# ORゲート
##################################################
def OR(x1, x2):
	
	x = np.array([[x1], [x2]]);
	w = np.array([[0.5, 0.5]]);
	b = -0.2
	
	y = np.dot(w, x) + b;
	
	if y > 0:
		return 1
	else:
		return 0
	
##################################################
# NANDゲート
##################################################
def NAND(x1, x2):
	
	x = np.array([[x1], [x2]]);
	w = np.array([[-0.5, -0.5]]);
	b = 0.9
	
	y = np.dot(w, x) + b;
	
	if y > 0:
		return 1
	else:
		return 0

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	print("hello, world")
	
	param = [[0,0],[0,1],[1,0],[1,1]]
	for x1, x2 in param:
		print( "AND (%.1f, %.1f) : %.1f" % (x1, x2, AND (x1, x2)) )
	
	for x1, x2 in param:
		print( "OR  (%.1f, %.1f) : %.1f" % (x1, x2, OR  (x1, x2)) )
	
	for x1, x2 in param:
		print( "NAND(%.1f, %.1f) : %.1f" % (x1, x2, NAND(x1, x2)) )
		
	#print( "AND (%.1f,%.1f) : %.1f" % (1.0, 1.0, AND (1.0, 1.0)) )
	#print( "OR  (%.1f,%.1f) : %.1f" % (1.0, 1.0, OR  (1.0, 1.0)) )
	#print( "NAND(%.1f,%.1f) : %.1f" % (1.0, 0.0, NAND(1.0, 1.0)) )
	

