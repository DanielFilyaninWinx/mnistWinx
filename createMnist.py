from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

if __name__ == '__main__':
	train = pd.read_csv("mnist_train.csv")# подгружаем данные для тр.
	Y = train['label']
	X = train.drop(['label'], axis=1)

	# Разделяем данные
	x_train, x_val, y_train, y_val = train_test_split(X.values, Y.values, test_size=0.18, random_state=1000)
	print(x_train.shape, y_train.shape)
	print(x_val.shape, y_val.shape)

	# размерность картинки
	img_rows, img_cols = 28, 28
	# преобразование выборок
	#x_train = x_train.reshape(60000, 784)#x_train = x_train.reshape(x_train.shape[0], , img_cols, 1)
	#x_val = x_val.reshape(10000, 784)
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
	# Нормализация данных
	x_train = x_train.astype('float32')
	x_val = x_val.astype('float32')
	# делим интенсивность каждого пикселя изображения на 255
	x_train /= 255
	x_val /= 255

	input_shape = (img_rows, img_cols, 1)

	#Преобразуем метки в формат one hot encoding
	y_train = keras.utils.to_categorical(y_train, 10)
	y_val = keras.utils.to_categorical(y_val, 10)

	# Создаем нейронную сеть
	# Создаем последовательную модель
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))

	# Компилируем сеть
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
	model.summary()

	# Обучаем нейронную сеть
	model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=1, validation_data=(x_val, y_val))

	#Оценка качества обучения
	scores = model.evaluate(x_val, y_val, verbose=1)
	print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 10))

	# Записываем полученную модель
	model.save('model.h5')
	print("Saved model to disk")

