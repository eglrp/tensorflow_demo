import numpy as np
import matplotlib.pyplot as plt
import cv2
import load
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#train_step = tf.train.MomentumOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

result = []
result_index = []

for i in range(1000): 
	index = np.random.uniform(0,60000,size=1000).astype(int)
	batch_xs = load.train_data[index]
	batch_ys = load.train_label[index]
	#print(batch_xs.shape)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	if i % 100 == 0:
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		ac = sess.run(accuracy, feed_dict={x: load.test_data, y_: load.test_label})
		result_index.append(i)
		result.append(ac)
		print(ac)
		
plt.plot(result_index, result,'r', label='accuracy')  
#plt.xticks(result, result, rotation=0) 
plt.grid()  
plt.show()  

if __name__ == '__main__':
	pass
	#saveimg()
	#test_data = getdata('test/',10000)
	#test_label = read_label('MNIST_data/t10k-labels.idx1-ubyte')
	#train_data = getdata('train/',60000)
	#train_label = read_label('MNIST_data/train-labels.idx1-ubyte')
