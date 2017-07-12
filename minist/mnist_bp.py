import numpy as np
import matplotlib.pyplot as plt
import cv2
import load
import tensorflow as tf

learning_rate = 0.01
training_epochs = 1500
batch_size = 2000
display_step = 1

n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
layer_1 = tf.nn.relu(layer_1)
# Hidden layer with RELU activation
layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
layer_2 = tf.nn.relu(layer_2)
# Output layer with linear activation
out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

pred = out_layer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

result = []
result_index = []

for epoch in range(training_epochs):
	#avg_cost = 0.
	index = np.random.uniform(0,60000,size=batch_size).astype(int)
	batch_xs = load.train_data[index]
	batch_ys = load.train_label[index]
	sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
	if epoch % 50 == 0:
		correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		ac = sess.run(accuracy, feed_dict={x: load.test_data, y: load.test_label})
		result_index.append(epoch)
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
