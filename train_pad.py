# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:40:41 2019

@author: cks
"""






import cv2
import numpy as np
import glob
import time
import os


no_gesture_dir = "C:/Users/cks/Videos/dataset/No Gesture"
swipe_right_dir = "C:/Users/cks/Videos/dataset/Swiping Right"
swipe_up_dir = "C:/Users/cks/Videos/dataset/Swiping Up"
drumming_fingers_dir = "C:/Users/cks/Videos/dataset/Drumming Fingers"
shaking_hand_dir = "C:/Users/cks/Videos/dataset/Shaking Hand"
other_gesture_dir = "C:/Users/cks/Videos/dataset/other"

dir_list = [no_gesture_dir,swipe_right_dir,swipe_up_dir,other_gesture_dir]

img_size = 28
frames = 20
stack_size = 0
for i in dir_list:
    stack_size += len(os.listdir(i))

input_layer = np.zeros((stack_size,frames,img_size,img_size,3),dtype = np.uint8)
    

labels = []



label_count = 0
count = 0


########## Collecting frames of all selected gestures from dir_list #####################

for current_dir in dir_list:

    for i in os.listdir(current_dir):
        
        os.chdir(os.path.join(current_dir,i))
        images = [cv2.imread(file) for file in glob.glob("*.jpg")]
    
    
    
        #Initiate the input layer 
        input_layer_temp = np.zeros((len(images),img_size,img_size,3),dtype = np.uint8)
        
        #Resize the 16 image frames from the list into 256x256
        for j in range(len(images)):
            
            images[j] = cv2.resize(images[j],(img_size,img_size))
            
            
        #Copy the images in the list into our input_layer matrix which can be fed into our 3D Conv function
        for j in range(len(images)):
        
            input_layer_temp[j] = images[j]    
        
                
        for j in range(frames):
            
            if(len(images) > 2*frames):
                input_layer[count,j,:,:,:] = input_layer_temp[2*j,:,:,:]
                
            if(len(images)<frames):
                if(j >= len(images)):
                    input_layer[count,j,:,:,:] = input_layer_temp[len(images)-1,:,:,:]
                else:
                    input_layer[count,j,:,:,:] = input_layer_temp[j,:,:,:]
    
            
            else :
                input_layer[count,j,:,:,:] = input_layer_temp[j,:,:,:]
            
        count = count + 1
        labels.append(label_count)
    
    label_count += 1
    











######## Structuring data to be fed into learning framework #########


labels = np.array(labels).reshape(-1,1)

X = input_layer
Y = labels

####### End of Data Structuring ##################



###### Splitting data into train and test sets ##########

X_train = []
X_test = []
Y_train = []
Y_test = []

for i in range(len(X)):
    
    if(i%20 == 0):
        X_test.append(X[i])
        Y_test.append(Y[i])
    else:
        X_train.append(X[i])
        Y_train.append(Y[i])



X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

###### End of Train and Test split #########
        
        

#### Extracting 100 random samples #######

random_try_train = np.zeros(100)

for i in range(100):
    
    random_try_train[i] = np.random.randint(0,len(X_train))

random_try_test = np.zeros(100)

for i in range(100):
    
    random_try_test[i] = np.random.randint(0,len(X_test))



#
############# Visualization  of the 100 random samples ###############        
#    
#
#
#
#
#for i in random_try_train:
#    
#    for j in range(frames):
#        cv2.imshow("frames",X_train[int(i),j,:,:,:])
#        cv2.waitKey(1)
#    time.sleep(0.1)
#cv2.destroyAllWindows()
#
#
#for i in random_try_test:
#    
#    for j in range(frames):
#        cv2.imshow("frames",X_test[int(i),j,:,:,:])
#        cv2.waitKey(1)
#    time.sleep(0.1)
#cv2.destroyAllWindows()
#
#
#    
#
#
######### End of Visualization ###########
#
#
#
#
#
#



################### Learning Part #######################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, MaxPooling3D




####### Model Specs ################

#3D filter/kernels and maxpooling window dimensions
kernal_size = 3
pooling_size = 3

#Create a an object of Sequential()
model = Sequential()

#Add one set of 3D convolution + MaxPooling to make the first layer
model.add(Conv3D(8, (kernal_size,kernal_size,kernal_size),padding = 'same', input_shape = X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size = (pooling_size,pooling_size,pooling_size),padding = 'same'))



model.add(Conv3D(16, (kernal_size,kernal_size,kernal_size),padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size = (pooling_size,pooling_size,pooling_size),padding = 'same'))

model.add(Conv3D(32, (kernal_size,kernal_size,kernal_size),padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size = (pooling_size,pooling_size,pooling_size),padding = 'same'))


model.add(Conv3D(64, (kernal_size,kernal_size,kernal_size),padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size = (pooling_size,pooling_size,pooling_size),padding = 'same'))

model.add(Conv3D(128, (kernal_size,kernal_size,kernal_size),padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size = (pooling_size,pooling_size,pooling_size),padding = 'same'))


model.add(Conv3D(256, (kernal_size,kernal_size,kernal_size),padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size = (pooling_size,pooling_size,pooling_size),padding = 'same'))

#model.add(Conv3D(512, (kernal_size,kernal_size,kernal_size),padding = 'same'))
#model.add(Activation('relu'))
#model.add(MaxPooling3D(pool_size = (pooling_size,pooling_size,pooling_size),padding = 'same'))





## Flatten the output of the final conv layer so that it can be fed to the fully connected layer
model.add(Flatten())

model.add(Dense(512))
                                                         
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(256))

model.add(Activation("relu"))
model.add(Dropout(0.5))

#Add the output layer with 2 neurons for the binary classifier(0 or 1)
model.add(Dense(4))
model.add(Activation("softmax"))

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])


model.summary()






# Pre-shuffle the data to introduce invariance while learning
from sklearn.utils import shuffle

X_train_sh, Y_train_sh = shuffle(X_train,Y_train)
#
##One hot encoding of labels
#from keras.utils.np_utils import to_categorical   
#Y_labels_sh = to_categorical(Y_train_sh, num_classes=6)

#Callbacks

#cb = TensorBoard()

#Start the training of the model

model.fit(X_train_sh,Y_train_sh,batch_size = 64, epochs= 23, validation_split = 0.1, shuffle = 'True')

########## Saving the trained model ###############.

model.save("C:/Users/cks/Documents/practice codes/final models/model_28x28_20f_19epochs_9.h5")





######### Loading saved model #####################


###Loading from saved model
#from keras.models import load_model
#
#model = load_model("C:/Users/cks/Documents/practice codes/gestures_3d_cnn_models/model_3_class_attempt1.h5")




#model.summary()
#model.get_weights()


Y_test_pred = model.predict(X_test)

Y_test_pred_classes = np.argmax(Y_test_pred,axis=1)
Y_test_pred_classes = Y_test_pred_classes.reshape(-1,1)


##### Shuffle the test set #############
from sklearn.utils import shuffle
X_test_sh, Y_test_sh = shuffle(X_test,Y_test)

##### Gesture Prediction on shuffled set ##########

Y_test_pred_sh = model.predict(X_test_sh)

Y_test_pred_classes_sh = np.argmax(Y_test_pred_sh,axis=1)
Y_test_pred_classes_sh = Y_test_pred_classes_sh.reshape(-1,1)



motion_list = ["No Gesture","Swipe Right ","Swipe Up ","Drumming Fingers","Shaking Hand","Other"]







## Performance % on the test sets ##############

(len(Y_test)-np.count_nonzero(Y_test - Y_test_pred_classes))/len(Y_test)

(len(Y_test_sh)-np.count_nonzero(Y_test_sh - Y_test_pred_classes_sh))/len(Y_test_sh)




