import numpy as np
import pandas as pd
import PIL
import matplotlib.pyplot as plt
import albumentations
from albumentations import (
    HorizontalFlip, Blur, MotionBlur, MedianBlur,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)


# вывести изображения и стоимость
def img_price_show(data, data_dir):
	# data - датафрейм с идентификатором и стоимостью автомобиля
	# data_dir - корень проекта

	plt.figure(figsize = (12,8))

	random_image = data.sample(n = 9)
	random_image_paths = random_image['sell_id'].values
	random_image_cat = random_image['price'].values

	for index, path in enumerate(random_image_paths):
	    im = PIL.Image.open(data_dir+'/img/img/' + str(path) + '.jpg')
	    plt.subplot(3, 3, index + 1)
	    plt.imshow(im)
	    plt.title('price: ' + str(random_image_cat[index]))
	    plt.axis('off')
	plt.show()


# вывести пример аугментации
def view_sample_augmentation(image, AUGMENTATIONS):
	# image - векторизованное изображение
	# AUGMENTATIONS - настройки аугментации

	plt.figure(figsize = (12,8))
	for i in range(9):
	    img = AUGMENTATIONS(image = image)['image']
	    plt.subplot(3, 3, i + 1)
	    plt.imshow(img)
	    plt.axis('off')
	plt.show()