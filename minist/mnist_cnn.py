import numpy as np
import matplotlib.pyplot as plt
import cv2
import load
import tensorflow as tf
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

learning_rate = 0.001
training_epochs = 2000
batch_size = 100
display_step = 10

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

result = []
result_index = []

for epoch in range(training_epochs):
	index = np.random.uniform(0,60000,size=batch_size).astype(int)
	batch_xs = load.train_data[index]
	batch_ys = load.train_label[index]
	sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
	#print(epoch)
	if epoch % 50 == 0:
		#correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		ac = sess.run(accuracy, feed_dict={x: load.test_data, y: load.test_label, keep_prob: dropout})
		result_index.append(epoch)
		result.append(ac)
		print(ac)

plt.plot(result_index, result,'r', label='accuracy')  
#plt.xticks(result, result, rotation=0) 
plt.grid()  
plt.show()
'''
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
with tf.name_scope('input') as scope:
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
with tf.name_scope('hiden_layer1') as scope:
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	tf.summary.histogram('hiden_layer1_W',weights['h1'])
	tf.summary.histogram('hiden_layer1_b',biases['b1'])
with tf.name_scope('hiden_layer2') as scope:
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	tf.summary.histogram('hiden_layer2_W',weights['h2'])
	tf.summary.histogram('hiden_layer2_b',biases['b2'])
with tf.name_scope('out') as scope:
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

pred = out_layer

with tf.name_scope('Loss') as scope:
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
with tf.name_scope('optimizer') as scope:
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

tf.summary.scalar("loss", cost)
	
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
merged = tf.summary.merge_all()  
summary_writer = tf.summary.FileWriter('board/', graph=sess.graph)

result = []
result_index = []

for epoch in range(training_epochs):
	#avg_cost = 0.
	index = np.random.uniform(0,60000,size=batch_size).astype(int)
	batch_xs = load.train_data[index]
	batch_ys = load.train_label[index]
	sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
	if epoch % 20 == 0:
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
'''
if __name__ == '__main__':
	pass
	#saveimg()
	#test_data = getdata('test/',10000)
	#test_label = read_label('MNIST_data/t10k-labels.idx1-ubyte')
	#train_data = getdata('train/',60000)
	#train_label = read_label('MNIST_data/train-labels.idx1-ubyte')
