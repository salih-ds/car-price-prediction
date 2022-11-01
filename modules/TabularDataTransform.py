import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# преобразовать табличные данные после EDA
def transform_tabular_base_data(data):
	# data - оригинальный датафрейм без изменений

	# bodyType - уберем информацию о дверях, объединим редкие данные с основными типами
	data.loc[data['bodyType'] == 'внедорожник 5 дв.', 'bodyType'] = 'внедорожник'
	data.loc[data['bodyType'] == 'хэтчбек 5 дв.', 'bodyType'] = 'хэтчбек'
	data.loc[data['bodyType'] == 'универсал 5 дв.', 'bodyType'] = 'универсал'
	data.loc[data['bodyType'] == 'хэтчбек 3 дв.', 'bodyType'] = 'хэтчбек'
	data.loc[data['bodyType'] == 'внедорожник 3 дв.', 'bodyType'] = 'внедорожник'
	data.loc[data['bodyType'] == 'седан 2 дв.', 'bodyType'] = 'седан'
	data.loc[data['bodyType'] == 'пикап двойная кабина', 'bodyType'] = 'внедорожник'
	data.loc[data['bodyType'] == 'внедорожник открытый', 'bodyType'] = 'внедорожник'
	data.loc[data['bodyType'] == 'компактвэн', 'bodyType'] = 'минивэн'

	# engineDisplacement - переведем в числовой формат, установим mean для undefined
	for i in range(len(data)):
	  data.engineDisplacement[i] = data.engineDisplacement[i][0:3]
	data.loc[data['engineDisplacement'] == 'und', 'engineDisplacement'] = 2.8
	data.engineDisplacement = pd.to_numeric(data.engineDisplacement, errors='coerce')

	# enginePower - приведем в числовой формат
	for i in range(len(data)):
	    data['enginePower'][i] = int(data['enginePower'][i][:-4])
	data.enginePower = pd.to_numeric(data.enginePower, errors='coerce')

	# Владельцы - добавим владельца для 1го nan
	data.loc[data.Владельцы.isnull(), 'Владельцы'] = '2\xa0владельца'

	# Владение - переведел в числовой формат (float) в годы, установил для nan среднее занчение в зависимости от ModelDate
	for n in range(len(data.Владение)):
	    if type(data.Владение[n]) == str:
	        line = data.Владение[n].split('и')
	    
	        if len(line) == 1:
	            # лет владения float
	            list_1 = []
	            # проверка, если месяцы
	            if 'месяц' in line[0]:
	                for i in line[0]:
	                    if '0' <= i <= '9':
	                        int_line = int(int(i)*8.3)
	                        list_1.append(float(f'0.{int_line}'))

	            else:
	                for i in line[0]:
	                    if '0' <= i <= '9':
	                        int_line = int(i)
	                        list_1.append(int_line)

	            data.Владение[n] = sum(list_1)

	        if len(line) == 2:
	            list_1 = []
	            for i in line[0]:
	                if '0' <= i <= '9':
	                    list_1.append(i)
	            list_1 = int(''.join(list_1))

	            list_2 = []
	            for i in line[1]:
	                if '0' <= i <= '9':
	                    list_2.append(i) 
	            list_2 = int(float(''.join(list_2))*8.3)

	            data.Владение[n] = float(f'{list_1}.{list_2}')
	        
	# Установим для nan среднее занчение для 10-ти и 5-ти летиям
	data_min_2000 = data.query('modelDate < 2000').Владение.mean()
	data_2000_2010 = data.query('modelDate >= 2000 & modelDate < 2010').Владение.mean()
	data_2010_2015 = data.query('modelDate >= 2010 & modelDate < 2015').Владение.mean()
	data_max_2015 = data.query('modelDate >= 2015').Владение.mean()

	data.loc[(data['modelDate'] >= 2000) & (data['modelDate'] < 2010) & (data['Владение'] != data['Владение']), 'Владение'] = data_2000_2010
	data.loc[(data['modelDate'] < 2000) & (data['Владение'] != data['Владение']), 'Владение'] = data_min_2000
	data.loc[(data['modelDate'] >= 2010) & (data['modelDate'] < 2015) & (data['Владение'] != data['Владение']), 'Владение'] = data_2010_2015
	data.loc[(data['modelDate'] >= 2015) & (data['Владение'] != data['Владение']), 'Владение'] = data_max_2015                                                                

	data.Владение = pd.to_numeric(data.Владение, errors='coerce')


# feauture enginering после первичного преобразования
def feauture_enginering_tab_data(data):
	# long, compact, competition, xDrive, AMG, Blue
	data['long'] = 0
	data['compact'] = 0
	data['competition'] = 0
	data['xdrive'] = 0
	data['amg'] = 0
	data['blue'] = 0

	# преобразуем в нижний регистр
	data['name'] = data['name'].str.lower()

	data.loc[(data['name'].str.contains("long")) | (data['name'].str.contains("длинн")), 'long'] = 1
	data.loc[(data['name'].str.contains("компак")), 'compact'] = 1
	data.loc[(data['name'].str.contains("competition")), 'competition'] = 1
	data.loc[(data['name'].str.contains("xdrive")), 'xdrive'] = 1
	data.loc[(data['name'].str.contains("amg")), 'amg'] = 1
	data.loc[(data['name'].str.contains("blue")), 'blue'] = 1

	# description_len - количество символов в описании
	data['description_len'] = 0
	for i in range(len(data)):
	  data['description_len'][i] = len(data.description[i])

	# разница даты производства и даты выхода моделей
	data['productionDate_minus_modelDate'] = data.productionDate - data.modelDate

	# разница даты производства и даты выхода моделей
	data['enginePower_on_engineDisplacement'] = data.enginePower / data.engineDisplacement

	# мощность двигателя на объем
	data['enginePower_on_engineDisplacement'] = data.enginePower / data.engineDisplacement

	# пробег на мощность двигателя
	data['mileage_on_enginePower'] = data.mileage / data.enginePower

	# пробег на объем двигателя
	data['mileage_on_engineDisplacement'] = data.mileage / data.engineDisplacement

	# пробег на срок владения автомобилем
	data['mileage_on_Владение'] = data.mileage / data.Владение

	# лет с момента выпуска модели
	productionDate_max = data.productionDate.max()
	data['productionDate_max_minus_modelDate'] = productionDate_max - data['modelDate']

	# лет с момента производства автомобиля
	data['productionDate_max_minus_productionDate'] = productionDate_max - data['productionDate']

	# ключевые слова из описания объявления, влияющие на стоимость
	data['торг'] = data.description.apply(lambda x: 1 if 'торг' in x else 0)
	data['срочн'] = data.description.apply(lambda x: 1 if 'срочн' in x else 0)
	data['скидк'] = data.description.apply(lambda x: 1 if 'скидк' in x else 0)
	data['шины'] = data.description.apply(lambda x: 1 if ('шин' in x) or ('резин' in x) else 0)
	data['подогрев'] = data.description.apply(lambda x: 1 if 'подогрев' in x or 'обогрев' in x else 0)
	data['диск'] = data.description.apply(lambda x: 1 if 'диск' in x else 0)
	data['скидк'] = data.description.apply(lambda x: 1 if 'скидк' in x else 0)
	data['не_битая'] = data.description.apply(lambda x: 1 if 'не бит' in x else 0)
	data['обмен'] = data.description.apply(lambda x: 1 if 'обмен' in x else 0)
	data['кредит'] = data.description.apply(lambda x: 1 if 'кредит' in x else 0)
	data['массаж'] = data.description.apply(lambda x: 1 if 'массаж' in x else 0)

	# миль на возраст автомобиля
	data['mileage_on_date'] = data['mileage'] / (data['productionDate_max_minus_productionDate']+1)


num_cols = ['engineDisplacement',
       'enginePower', 'mileage', 'modelDate',
       'numberOfDoors', 'productionDate', 'Владение',
       'long', 'compact', 'competition',
       'xdrive', 'amg', 'blue', 'description_len',
       'productionDate_minus_modelDate', 'enginePower_on_engineDisplacement',
       'mileage_on_enginePower', 'mileage_on_engineDisplacement',
       'mileage_on_Владение',
       'productionDate_max_minus_modelDate',
       'productionDate_max_minus_productionDate',
       'mileage_on_date']

columns = ['bodyType', 'productionDate', 'brand', 'model_info', 'color', 'engineDisplacement',
       'enginePower', 'fuelType', 'mileage',
       'numberOfDoors', 'mileage_on_enginePower',
       'vehicleTransmission', 'Владельцы', 'Владение', 'ПТС', 'Привод',
       'sample', 'price', 'long', 'compact', 'competition',
       'xdrive', 'amg', 'blue', 'description_len',
       'productionDate_minus_modelDate', 'enginePower_on_engineDisplacement',
       'mileage_on_engineDisplacement',
       'mileage_on_Владение',
       'productionDate_max_minus_modelDate',
       'торг', 'срочн', 'скидк',
       'шины', 'подогрев', 'диск', 'не_битая', 'обмен', 'кредит', 'массаж',
       'mileage_on_date']

cat_cols = ['bodyType', 'color', 'fuelType',
       'vehicleTransmission', 'Владельцы', 'ПТС', 'Привод', 'brand', 'model_info',]

def create_df(data, num_cols=num_cols, cat_cols=cat_cols, columns=columns):
	# устраним выбросы для числовых признаков
	data[num_cols] = pd.DataFrame(RobustScaler().fit_transform(data[num_cols]), columns = data[num_cols].columns)
	# оставим нужные колонки
	data = data[columns]
	# dummy-переменные
	data = pd.get_dummies(data, prefix=cat_cols, columns=cat_cols) 

	return(data)






