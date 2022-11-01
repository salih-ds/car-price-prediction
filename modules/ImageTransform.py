import numpy as np
import pandas as pd
import cv2
import albumentations
from albumentations import (
    HorizontalFlip, Blur, MotionBlur, MedianBlur, IAAAdditiveGaussianNoise,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,
    GaussNoise, CLAHE
)

# векторизовать изображения
def get_image_array(data, index, data_dir, size=(320, 240)):
    images_train = []
    for index, sell_id in enumerate(data['sell_id'].iloc[index].values):
        image = cv2.imread(data_dir + '/img/img/' + str(sell_id) + '.jpg')
        assert(image is not None)
        image = cv2.resize(image, size)
        images_train.append(image)
    images_train = np.array(images_train)
    print('images shape', images_train.shape, 'dtype', images_train.dtype)
    return(images_train)


# настройки для аугментации
def custom_augmentation():
	# оставим преобразования, которые не портят изображения
	AUGMENTATIONS = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5), 
    # albumentations.Rotate(limit=30, interpolation=1, border_mode=4,
                          # always_apply=False, p=0.5), 
    albumentations.OneOf([
        albumentations.CenterCrop(height=224, width=200),
        albumentations.CenterCrop(height=200, width=224),
    ], p=0.3), 
    albumentations.OneOf([
        albumentations.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3),
        albumentations.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1)
    ], p=0.5), 
    OneOf([
        MotionBlur(p=0.3),
        MedianBlur(blur_limit=3, p=0.1),
        Blur(blur_limit=3, p=0.1),
    ], p=0.3), 
    OneOf([
        IAAAdditiveGaussianNoise(),
        GaussNoise(),
    ], p=0.2), 
    OneOf([
        CLAHE(clip_limit=2),
        IAASharpen(),
        IAAEmboss(),
        RandomBrightnessContrast(),
    ], p=0.3),
    albumentations.Resize(240, 320)
	])

	return(AUGMENTATIONS)