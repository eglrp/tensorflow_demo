import tf_load
import numpy as np
import tensorflow as tf
import cv2

def weight_init(shape):
    weights = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(weights)
# 偏置的初始化
def biases_init(shape):
    biases = tf.random_normal(shape,dtype=tf.float32)
    return tf.Variable(biases)
# 全连接层权重初始化函数xavier
def xavier_init(layer1, layer2, constant = 1):
    Min = -constant * np.sqrt(6.0 / (layer1 + layer2))
    Max = constant * np.sqrt(6.0 / (layer1 + layer2))
    return tf.Variable(tf.random_uniform((layer1, layer2), minval = Min, maxval = Max, dtype = tf.float32))
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
train_epochs = 1000    # 训练轮数
batch_size   = 2000     # 随机出去数据大小
learning_rate= 0.0001  # 学习效率
drop_prob    = 0.75     # 正则化,丢弃比例
fch_nodes    = 100     # 全连接隐藏层神经元的个数

input_size = 28
output_size = 2
c_size = 3
c_layer1_deep = 6
c_layer2_deep = 12

x = tf.placeholder(tf.float32, [None, input_size*input_size*3])
y = tf.placeholder(tf.float32, [None, output_size])
x_image = tf.reshape(x, [-1, input_size, input_size, 3])
#print(np.shape(x_image[1]))

# 第一层卷积+池化
w_conv1 = weight_init([c_size, c_size, 3, c_layer1_deep])                             # 5x5,深度为1,16个
b_conv1 = biases_init([c_layer1_deep])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)    # 输出张量的尺寸:28x28x16
h_pool1 = max_pool_2x2(h_conv1)  
# 第二层卷积+池化
w_conv2 = weight_init([c_size, c_size, c_layer1_deep, c_layer2_deep])                             # 5x5,深度为16,32个
b_conv2 = biases_init([c_layer2_deep])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)    # 输出张量的尺寸:14x14x32
h_pool2 = max_pool_2x2(h_conv2) 

# 全连接层
h_fpool2 = tf.reshape(h_pool2, [-1, 7*7*c_layer2_deep])
# 全连接层,隐藏层节点为512个
# 权重初始化
w_fc1 = xavier_init(7*7*c_layer2_deep, fch_nodes)
b_fc1 = biases_init([fch_nodes])
h_fc1 = tf.nn.relu(tf.matmul(h_fpool2, w_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=drop_prob)

w_fc2 = xavier_init(fch_nodes, 2)
b_fc2 = biases_init([2])

# 未激活的输出
y_ = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)
# 激活后的输出
y_out = tf.nn.softmax(y_)

# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) 
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=y))
cost = -tf.reduce_sum(y*tf.log(y_out))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()



sess.run(init)
for epoch in range(train_epochs):
	index = np.random.uniform(0,8400,size=batch_size).astype(int)
	batch_xs = tf_load.train_data[index]
	batch_ys = tf_load.train_label[index]
	sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
	#if epoch % 5 == 0:
	ac,c = sess.run([accuracy,cost], feed_dict={x: batch_xs, y: batch_ys})
	print(str(ac)+'---------'+str(c))
	