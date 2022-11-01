import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# грфик ошибки по эпохам
def history_loss_metrics(history):
	# history - история обучения сети

	plt.title('Loss')
	plt.plot(history.history['MAPE'], label='train')
	plt.plot(history.history['val_MAPE'], label='test')
	plt.legend()
	plt.show();


# MAPE
def mape(y_true, y_pred):
	# y_true - правильные ответы для тестовых данных
	# y_pred - предсказанные ответы для тестовых данных (для неросети использовать y_pred[:,0])

	print(f"TEST mape:{np.mean(np.abs((y_pred-y_true)/y_true))*100:0.2f}%")