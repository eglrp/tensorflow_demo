import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import csv

def read_image(filename,path):
	f = open(filename, 'rb')
	index = 0
	buf = f.read()
	f.close()
	magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
	index += struct.calcsize('>IIII')

	for i in range(images):
		#for i in xrange(2000):
		image = Image.new('L', (columns, rows))

		for x in range(rows):
			for y in range(columns):
				image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
				index += struct.calcsize('>B')


		print('save ' + str(i) + 'image')
		image.save(path+'/' + str(i) + '.png')
	print('搞定')

def read_label(filename):
	result = []

	f = open(filename, 'rb')
	index = 0
	buf = f.read()
	f.close()

	magic, labels = struct.unpack_from('>II' , buf , index)
	index += struct.calcsize('>II')
  
	labelArr = [0] * labels

	for x in range(labels):
		#for x in xrange(2000):
		labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
		index += struct.calcsize('>B')
		#print(labelArr[x])
		num = labelArr[x]
		label = [0,0,0,0,0,0,0,0,0,0]
		label[num] = 1
		result.append(label)
		
	return result

# 把minist数据集转换为图像格式，第一次运行的时候使用
def saveimg():
	read_image('MNIST_data/train-images.idx3-ubyte','train')
	#read_label('MNIST_data/train-labels.idx1-ubyte', 'train/label.txt')
	#read_image('MNIST_data/t10k-images.idx3-ubyte','test')
	#read_label('MNIST_data/t10k-labels.idx1-ubyte', 'test/label.txt')
def getdata(root,size):
	data = []
	for i in range(size):
		filename = str(i)+'.png'
		img = cv2.imread(root+filename, 0)
		img = img.reshape(28*28)
		data.append(img)
	return data

test_data = np.array(getdata('test/',10000))
test_label = np.array(read_label('MNIST_data/t10k-labels.idx1-ubyte'))
train_data = np.array(getdata('train/',60000))
train_label = np.array(read_label('MNIST_data/train-labels.idx1-ubyte'))

if __name__ == '__main__':
	pass
	#saveimg()
	#test_data = getdata('test/',10000)
	#test_label = read_label('MNIST_data/t10k-labels.idx1-ubyte')
	#train_data = getdata('train/',60000)
	#train_label = read_label('MNIST_data/train-labels.idx1-ubyte')
