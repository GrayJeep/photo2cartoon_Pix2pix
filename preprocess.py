# coding:UTF-8
import cv2
import numpy as np
import os
import tensorflow as tf

train_A_path = 'dataset/photo2cartoon/train_A/'  # 训练集真人头像
trainB_path = 'dataset/photo2cartoon/trainB/'  # 训练集卡通头像
test_A_path = 'dataset/photo2cartoon/test_A/'  # 测试集真人头像
testB_path = 'dataset/photo2cartoon/testB/'  # 测试集卡通头像

# 训练集
trainA = []
trainB = []
# 测试集
testA = []
testB = []
batch_size = 32

for filename in os.listdir(train_A_path):
    file_path = train_A_path + filename
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    # 归一化
    img = img.astype('float32') / 255
    trainA.append(img)

for filename in os.listdir(trainB_path):
    file_path = trainB_path + filename
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255
    trainB.append(img)

x = np.array(trainA)
y = np.array(trainB)
# 构建数据集对象
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.batch(batch_size)

# 测试集
for filename in os.listdir(test_A_path):
    file_path = test_A_path + filename
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255
    testA.append(img)

for filename in os.listdir(testB_path):
    file_path = testB_path + filename
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255
    testB.append(img)

test_x = np.array(testA)
test_y = np.array(testB)
