# -*- coding: utf-8 -*-
"""

"""


# Basic Libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Evaluation library
from sklearn.metrics import confusion_matrix


from tensorflow.keras import layers
import keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from tensorflow.keras.utils import to_categorical


#Fashion MNIST dataset
letter_train=pd.read_csv(r"E:\study\year3\AI\dataset\emnist-letters-train.csv")
letter_test=pd.read_csv(r"E:\study\year3\AI\dataset\emnist-letters-test.csv")


x_train = letter_train.iloc[:, 1:]
y_train = letter_train.iloc[:, 0]
x_test = letter_test.iloc[:, 1:]
y_test = letter_test.iloc[:, 0]
x_train, x_test = x_train / 255.0, x_test/ 255.0


y_train_digit = to_categorical(y_train, num_classes=122)

y_test_digit = to_categorical(y_test, num_classes=122)

#Creating base neural network
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(122,activation='sigmoid'),
])

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics = ['accuracy'])

model.fit(x_train, y_train_digit, batch_size=100, epochs=20)

test_loss_digit, test_acc_digit = model.evaluate(x_test, y_test_digit)

#Predicting the labels-DIGIT
y_predict = model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1) # Here we get the index of maximum value in the encoded vector
y_test_digit_eval=np.argmax(y_test_digit, axis=1)


#Confusion matrix for Digit MNIST
con_mat=confusion_matrix(y_test_digit_eval,y_predict)
plt.style.use('seaborn-deep')
plt.figure(figsize=(10,10))
sns.heatmap(con_mat,annot=True,annot_kws={'size': 15},linewidths=0.5,fmt="d",cmap="gray")
plt.title('True or False predicted digit MNIST\n',fontweight='bold',fontsize=15)
plt.show()



