from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from tensorflow import keras
import PIL
import glob
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing import image

import os

img_width, img_height = 28, 28
input_shape = (img_width, img_height, 1)

if __name__ == '__main__':

	model = keras.models.load_model('model.h5')
	os.chdir(os.getcwd() + '/test')
	print( os.getcwd() )
	# Загружаем свою картинку
	name = '1.jpg'
	img_path = input('test//' + name)
	img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")

	plt.imshow(img.convert('RGBA'))
	plt.show()

	# Преобразуем картинку в массив
	x = image.img_to_array(img)
	# Меняем форму массива в плоский вектор
	x = x.reshape(1, 784)
	# Инвертируем изображение
	x = 255 - x
	# Нормализуем изображение
	x /= 255

	prediction = model.predict(x)
	prediction
	prediction = np.argmax(prediction)
	print("ответ:", prediction)