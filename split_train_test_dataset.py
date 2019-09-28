from glob import glob
import os
import numpy as np
import random

dataset_root = '/home/gaojw/AI/ECG-MIT-BIH-Arrythmia/mit-bih-2D_personal_program/MIT-BIH_AD/'
output_dirs = ['NOR/', 'LBBB/', 'RBBB/', 'APC/', 'PVC/', 'PAB/', 'VEB/', 'VFE/']

count = 0
pathes_by_type = {}
for type in output_dirs:
	dir = os.path.join(dataset_root, type, '*.png')
	paths = glob(dir)
	pathes_by_type[type] = paths
	count += len(paths)

train_list = []
val_list = []
test_list = []

for type in output_dirs:
	cur = pathes_by_type[type]
	if len(cur) is 0:
		continue		#跳过类别数量为0的分类
	random.shuffle(cur)		#随机打乱各类别中的样本顺序
	#抽调60%的数据作为train set
	for i in range(int(len(cur)*0.6)):
		temp=cur[i].split('/')
		train_list.append('{} {}'.format(os.path.join(temp[-2], temp[-1]), output_dirs.index(type)))    #output_dirs.index(type) 是类别的索引
		#建立一种格式  file_name label   eg. Normal/fig_21471.png 0
		cur[i]=None

	#抽调200%的数据作为val set
	for i in range(int(len(cur)*0.6), int(len(cur)*0.8)):
		if cur[i] is None:
			continue
		else:
			temp = cur[i].split('/')
			val_list.append('{} {}'.format(os.path.join(temp[-2], temp[-1]), output_dirs.index(type)))
			cur[i] = None
	#抽调20%的数据作为test set
	for i in range(int(len(cur) * 0.8), len(cur)):
		if cur[i] is None:
			continue
		else:
			temp = cur[i].split('/')
			test_list.append('{} {}'.format(os.path.join(temp[-2], temp[-1]), output_dirs.index(type)))
			cur[i] = None

	#generate the txt_file which record the png_file and it's label belong to 
	with open('MIT-BIH_AD_val.txt', 'w') as val:
		for v in val_list:
			val.write(v+'\n')

	with open('MIT-BIH_AD_train.txt', 'w') as train:
		for r in train_list:
			train.write(r+'\n')

	with open('MIT-BIH_AD_test.txt', 'w') as test:
		for t in test_list:
			test.write(t+'\n')

print('train:{} val:{} test:{} tol:{}'.format(len(train_list), len(val_list), len(test_list), count))