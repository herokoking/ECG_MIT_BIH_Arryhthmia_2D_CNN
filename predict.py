#!/home/gaojw/src/python3/python3/bin/python3
from model import proposed_model
from keras.optimizers import Adam
import numpy as np
import cv2
import os
import time
import random
import glob
import os
from keras.utils import np_utils
from tqdm import tqdm

class_names = ['NOR','LBBB', 'RBBB', 'APC', 'PVC', 'PAB', 'VEB', 'VFE']

imageh = 128
imagew = 128

inputH = 128
inputW = 192

#build model and load  trained weights
model = proposed_model()
lr = 0.0001
adm = Adam(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
#model.summary()

model.load_weights('./result/mit_bih_2D.hdf5', by_name=True)

#load test dataset 
test_file = './MIT-BIH_AD_test.txt'
test_img_path = './MIT-BIH_AD/'
augmentation = True
output_img = False

f = open(test_file, 'r')
lines = f.readlines()
random.shuffle(lines)
TP = 0
count = 0
total = len(lines)
#创造两个空字典，用以记录各个类别的counter and tp_counter 
counter = {'NOR':0,'LBBB': 0, 'RBBB': 0, 'APC': 0, 'PVC': 0, 'PAB': 0, 'VEB': 0, 'VFE': 0}
tp_counter = {'NOR':0,'LBBB': 0, 'RBBB': 0, 'APC': 0, 'PVC': 0, 'PAB': 0, 'VEB': 0, 'VFE': 0}


for line in tqdm(lines):
	path = line.split(' ')[0]
	label = line.split(' ')[-1]

	label = label.strip('\n')
	answer = int(label)
	img = os.path.join(test_img_path, path)
	#eg.  img='./MIT-BIH_AD/NOR/fig_39572.png'
	image = cv2.imread(img)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	if augmentation:
		'''
		如果采用数据增强，则平移缩放图片
		'''
		Hshmean = int(np.round(np.max([0, np.round((imageh - inputH) / 2)])))
		Wshmean = int(np.round(np.max([0, np.round((imagew - inputW) / 2)])))
		image = image[Hshmean:Hshmean + inputH, Wshmean:Wshmean + inputW, :]
		image = cv2.resize(image, (imagew, imageh))
	else:  
		pass
	#逐个png文件预测
	input_data = np.zeros((1, imagew, imageh, 3), dtype='float32')
	input_data = image.reshape(1,128,128,3)
	pred = model.predict(input_data)
	label = np.argmax(pred)
	#TP means for true positive
	if label == answer:
		TP += 1
		tp_counter[class_names[label]] += 1
	count += 1
	counter[class_names[answer]] += 1

print('Total:    Acc = {} '.format(str(TP / count)))
print('LBBB:{}/{}={},\n RBBB:{}/{}={},\n APC:{}/{}={},\n PVC:{}/{}={},\n PAB:{}/{}={},\n VEB:{}/{}={},\n VFE:{}/{}={}'.format(
	tp_counter['LBBB'], counter['LBBB'], (tp_counter['LBBB'] / counter['LBBB']),
	tp_counter['RBBB'], counter['RBBB'], (tp_counter['RBBB'] / counter['RBBB']),
	tp_counter['APC'], counter['APC'], (tp_counter['APC'] / counter['APC']),
	tp_counter['PVC'], counter['PVC'], (tp_counter['PVC'] / counter['PVC']),
	tp_counter['PAB'], counter['PAB'], (tp_counter['PAB'] / counter['PAB']),
	tp_counter['VEB'], counter['VEB'], (tp_counter['VEB'] / counter['VEB']),
	tp_counter['VFE'], counter['VFE'], (tp_counter['VFE'] / counter['VFE'])
	))
