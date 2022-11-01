import numpy as np
import pandas as pd

def primary_info_cols(data, cols):
	# data - датафрейм
	# cols - список названий столбцов

	for col in cols:
		print(f'{col}')
		print()
		print('unique:')
		print(data[col].unique())
		print()
		print('value_counts:')
		print(data[col].value_counts())
		print()
		print('-'*50)
		print()