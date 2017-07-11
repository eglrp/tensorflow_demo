import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
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

def read_label(filename, saveFilename):
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
		
	save = open(saveFilename, 'w')

	save.write(','.join(map(lambda x: str(x), labelArr)))
	save.write('\n')

	save.close()
	print('save labels success')

# 把minist数据集转换为图像格式，第一次运行的时候使用
def saveimg():
	read_image('MNIST_data/train-images.idx3-ubyte','train')
	read_label('MNIST_data/train-labels.idx1-ubyte', 'train/label.txt')
	read_image('MNIST_data/t10k-images.idx3-ubyte','test')
	read_label('MNIST_data/t10k-labels.idx1-ubyte', 'test/label.txt')
if __name__ == '__main__':
	pass