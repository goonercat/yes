import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

data = tf.keras.datasets.mnist
(xtr,ytr),(xte,yte) = data.load_data()

xtr = tf.keras.utils.normalize(xtr, axis=1)
xte = tf.keras.utils.normalize(xte, axis=1)
'''
mdl = tf.keras.models.Sequential()
mdl.add(tf.keras.layers.Flatten(input_shape=(28,28)))
mdl.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
mdl.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
mdl.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
mdl.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))#softmax ensures the output is between 0 and 1 and the sum of all outputs is 1

mdl.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

mdl.fit(xtr,ytr,epochs=4)#epochs is the number of times the model will see the entire training data

mdl.save('digrec.keras')
'''
mdl = tf.keras.models.load_model('digit recognition/handwritten.keras')

loss, acc = mdl.evaluate(xte,yte)
print('loss: ', loss)
print('accuracy: ', acc)

os.path.isfile("digit recognition/1234.png")
img = cv2.imread("digit recognition/1234.png")[:,:,0]#grayscale
img = np.invert(np.array([img]))
prediction = mdl.predict(img)
print("predicted digit: ", np.argmax(prediction))
plt.imshow(img[0], cmap=plt.cm.binary)
