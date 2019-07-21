#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
#!pip3 install -q tensorflow-gpu==2.0.0-beta0
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam


# In[ ]:


# Block 1
model = models.Sequential()
model.add(layers.ZeroPadding2D((1,1),input_shape=(224,224,4)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))


# In[ ]:


# Block 2
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D (128, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))


# In[ ]:


# Block 3
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))


# In[ ]:


# Block 4
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))


# In[ ]:


# Block 5
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))


# In[ ]:


# # Classification block
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


import sys, os, glob, shutil, re
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy


# In[ ]:


train_lbls = []
train_data = []

train_zero_index = []
train_one_index = []
train_index = 0

for img in os.listdir("/Users/junzhejiang/Desktop/umtri/reporportion_train/train"):
    if re.findall(r'.png$',img) != []:
        image = Image.open('/Users/junzhejiang/Desktop/umtri/reporportion_train/train/'+img)
        new_img = image.resize((224,224))
        new_img.save("/Users/junzhejiang/Desktop/umtri/reporportion_train/train/resized.png", optimize=True)
        train_data.append(mpimg.imread("/Users/junzhejiang/Desktop/umtri/reporportion_train/train/resized.png"))
        # train_data.append(mpimg.imread('./CNN_static_data/Train/'+img))
        if re.findall(r'1\d{3}.png$',img) != []:
            train_lbls.append(1)
            train_one_index.append(train_index)
        else:
            train_lbls.append(0)
            train_zero_index.append(train_index)
        train_index += 1
            
            
train_lbls = np.array(train_lbls) 
train_data = np.array(train_data,dtype=np.float64)
print(train_data.shape)



# In[ ]:


test_lbls = []
test_data = []

test_zero_index = []
test_one_index = []
test_index = 0

for img in os.listdir("/Users/junzhejiang/Desktop/umtri/reporportion_train/test"):
    if re.findall(r'.png$',img) != []:
        image = Image.open('/Users/junzhejiang/Desktop/umtri/reporportion_train/test/'+img)
        new_img = image.resize((224,224))
        new_img.save("/Users/junzhejiang/Desktop/umtri/reporportion_train/test/resized.png", optimize=True)
        test_data.append(mpimg.imread("/Users/junzhejiang/Desktop/umtri/reporportion_train/test/resized.png"))
        # test_data.append(mpimg.imread('./CNN_static_data/Test/'+img))
        if re.findall(r'1\d{3}.png$',img) != []:
            test_lbls.append(1)
            test_one_index.append(test_index)
        else:
            test_lbls.append(0)
            test_zero_index.append(test_index)
        test_index += 1
            
test_lbls = np.array(test_lbls) 
test_data = np.array(test_data,dtype=np.float64)
print( test_data.shape)


# In[ ]:


# Normalize pixel values to be between 0 and 1
train_data, test_data = train_data / 255.0, test_data / 255.0


# In[ ]:


np.mean( test_lbls[test_zero_index[:len(test_one_index)] + test_one_index]  )


# In[ ]:


model.compile(optimizer= Adam(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data[train_zero_index[:len(train_one_index)] + train_one_index] , 
          train_lbls[train_zero_index[:len(train_one_index)] + train_one_index] , 
          epochs=5)


# In[ ]:


test_loss, test_acc = model.evaluate(test_data[test_zero_index[:len(test_one_index)] + test_one_index] , 
                                     test_lbls[test_zero_index[:len(test_one_index)] + test_one_index] )


# In[ ]:


print( model.predict_classes(test_data[test_zero_index[:len(test_one_index)] + test_one_index] ) )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



