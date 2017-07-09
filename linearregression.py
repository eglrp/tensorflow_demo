import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
r_data = np.float32(np.random.rand(100))/20
z_data = np.dot([0.100, 0.200], x_data) + 0.300 + r_data

# 画出散点图分布
#ax = plt.figure().add_subplot(111, projection = '3d') 
#ax.scatter(x_data[0], x_data[1], z_data, c = 'r', marker = '^')
#ax.set_xlabel('X Label')  
#ax.set_ylabel('Y Label')  
#ax.set_zlabel('Z Label')  
#plt.show()

# 回归参数
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# 回归方程
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - z_data))
# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化
init = tf.initialize_all_variables()

# 运行
sess = tf.Session()
sess.run(init)
var_w = 0
var_b = 0
for step in range(0, 201):
	sess.run(train)
	if step % 20 == 0:
		#print(step, sess.run(W), sess.run(b))
		var_w = sess.run(W)[0]
		var_b = sess.run(b)[0]
		print('step='+str(step))
		print('w   ='+str(var_w))
		print('b   ='+str(var_b))
#x_data = np.matrix(x_data).T
#var_w = np.matrix(var_w).T
x = np.arange(0, 1, 0.1)  
y = np.arange(0, 1, 0.1)  
x, y = np.meshgrid(x, y)  
input = np.vstack((x.reshape(100),y.reshape(100))).T
result = np.dot(input,var_w) + var_b
result = result.reshape(10,10)
#print(result.shape)

# 画出散点图分布
ax = plt.figure().add_subplot(111, projection = '3d') 
ax.scatter(x_data[0], x_data[1], z_data, c = 'r', marker = '^')
ax.set_xlabel('X Label')  
ax.set_ylabel('Y Label')  
ax.set_zlabel('Z Label')  
ax.plot_surface(x, y, result,label='',alpha=0.3)
plt.show()