# car-price-prediction
Прогнозирование стоимости автомобиля по характеристикам, описанию и фотографии

Соревнование kaggle (Rus_Salih_b) Top-12%: https://www.kaggle.com/competitions/sf-dst-car-price-prediction-part2

## Требования
Python 3.7.6
Зависимости: requirements.txt

## Данные
https://www.kaggle.com/competitions/sf-dst-car-price-prediction-part2/data

## Обзор
### Tabular Data
Работаем с табличными данными

Посмотрим на типы признаков:
- bodyType - категориальный
- brand - категориальный
- color - категориальный
- description - текстовый
- engineDisplacement - числовой, представленный как текст
- enginePower - числовой, представленный как текст
- fuelType - категориальный
- mileage - числовой
- modelDate - числовой
- model_info - категориальный
- name - категориальный, желательно сократить размерность
- numberOfDoors - категориальный
- price - числовой, целевой
- productionDate - числовой
- sell_id - изображение (файл доступен по адресу, основанному на sell_id)
- vehicleConfiguration - не используется (комбинация других столбцов)
- vehicleTransmission - категориальный
- Владельцы - категориальный
- Владение - числовой, представленный как текст
- ПТС - категориальный
- Привод - категориальный
- Руль - категориальный

#### EDA
**Посмотрим на данные и преобразуем в нужный формат:**

    transform_tabular_base_data(data=data)
- bodyType - Убрал информацию о дверях (тк есть отдельный признак), объединил редкие значения с похожими типами
- description - вытащить число символов описания, использовать признак для nlp
- engineDisplacement - перевел в числовой формат, установил mean для undefined
- enginePower - убрал текстовое обозначение, приведел в числовой формат
- name - выделить отдельным признаком long, compact, competition, xDrive, AMG, Blue
- vehicleConfiguration - удалить столбец, т.к. дублирует существующие признаки
- Владение - переведел в числовой формат (float) в годы, установил для nan среднее занчение в зависимости от ModelDate
- Руль - удалить, т.к. почти 100% слева

#### Feauture Enginering

    feauture_enginering_tab_data(data=data)
- long, compact, competition, xDrive, AMG, Blue - характеристики из name
- description_len - количество символов в описании
- productionDate_minus_modelDate - разница даты производства и даты выхода моделей
- enginePower_on_engineDisplacement - мощность двигателя на объем
- mileage_on_enginePower - пробег на мощность двигателя
- mileage_on_engineDisplacement - пробег на объем двигателя
- mileage_on_Владение - пробег на срок владения автомобилем
- productionDate_max_minus_modelDate - лет с момента выпуска модели
- productionDate_max_minus_productionDate - лет с момента производства автомобиля
- key_words_description - ключевые слова из описания объявления, влияющие на стоимость
- mileage_on_date - миль на возраст автомобиля

####  Анализ корреляций признаков
<img width="1406" alt="corr-1" src="https://user-images.githubusercontent.com/73405095/198285885-b6949f74-6d8d-45c0-bf7f-39a8fb7e90aa.png">

- engineDisplacement и enginePower 0.9, enginePower имеет большую корреляцию с таргет. Оставляем оба, тк высокая корреляция с таргетом.
- mileage и modelDate/productionDate -0.7, mileage_on_enginePower/mileage_on_engineDisplacement 0.9, mileage_on_productionDate_norm100 1, productionDate_max_minus_modelDate/productionDate_max_minus_productionDate 0,7. 
- modelDate и productionDate 1, productionDate_max_minus_modelDate/productionDate_max_minus_productionDate -1.

**mileage_on_productionDate_norm100, mileage_on_enginePower, modelDate, productionDate, productionDate_max_minus_productionDate - удалены из-за высоко корреляции с признаками и меньшей корреляции с целевой переменной - в конечный код не добавлены!**

#### Подготовим датафрейм на вход модели
Устраним выбросы для числовых признаков, кодируем категориальный признаки и составим датафрейм

    df = create_df(data=data)
      
#### Обучим модель на табличных данных
Перемещаем и разобьем данные на тренировочную и валидационную выборки в пропорции 85/15
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=RANDOM_SEED)
    
Протестируем модели:
<table>
  <tbody><tr>
    <th>Модель</th>
    <th>Описание</th>
    <th>best MAPE</th>
    <th>Комментарий</th>
  </tr>
    <tr>
    <td>Neural Network 1 (Simple)</td>
    <td>Базовая сеть - классический персептрон</td>
    <td>10.55%</td>
    <td>Лучший результат</td>
  </tr>
  <tr>
    <td>Neural Network 2 (Relu to LeakyRelu)</td>
    <td>Снизу используем слой LeakyReLU</td>
    <td>10.62%</td>
    <td>Резульат хуже базового</td>
  </tr>
  <tr>
    <td>Neural Network 3 (bottle neck)</td>
    <td>Сеть по принципу bottle neck</td>
    <td>10.81%</td>
    <td>Резульат хуже базового</td>
  </tr>
  <tr>
    <td>CatBoost</td>
    <td>Подбор гиперпараметров с помощью gridsearch не помог, логарифмирование y улучшило метрику</td>
    <td>11.17%</td>
    <td>Достигли лучшего результата для CatBoost</td>
  </tr>
  <tr>
    <td>Neural Network 1 (Simple) - optimize</td>
    <td>Используем веса базовой сети и переобучим, снизив шаг спуска</td>
    <td>10.44%</td>
    <td>Достигли лучшего результата</td>
  </tr>
</tbody></table>

Neural Network 1 (Simple) достигла лучшего результата по метрике
<img width="552" alt="simple_network" src="https://user-images.githubusercontent.com/73405095/198872126-83580296-c2a5-4791-b14b-3b363c92155a.png">

**Сохраним обученную сеть Neural Network 1 (Simple) и CatBoost**

### Tabular + NLP
Создадим модель с подачей на вход табличных данных и текстовых (description)

В качестве признака оставляем только описание объявления

    data_nlp = data_nlp[['description', 'sample', 'price']]
    
Проведем очистку данных:
- приведем все символы в нижний регистр
- оставим только числа и русские и английские символы
- удалим стоп-слова
- проведем лемматизацию

Подготовим данные на вход модели, протестируем методы:
- bag of words
- 2 gram (словосочетание из 2-х слов)

#### Построим и обучим Multi-Input сеть: Tabular + Text
На вход сеть принимает отдельно табличные данные и текстовые, и соединяется в голове

Составим базовую сеть
<img width="833" alt="concat_mlp_nlp" src="https://user-images.githubusercontent.com/73405095/198941802-0efcde81-2061-4a8c-bd5d-5c49dc34407e.png">


Определим лучший метод токенизации, обучив сеть на разных данных
- bag of words, mape: 10.64%
- 2 gram, mape: 10.72%
**bag of words имеет лучший результат**

Протестируем различные архитектуры для NLP:
<table>
  <tbody><tr>
    <th>Модель</th>
    <th>Описание</th>
    <th>best MAPE</th>
    <th>Комментарий</th>
  </tr>
  <tr>
    <td>Base</td>
    <td>Постепенно снижение количества нейронов в слоях ближе к голове, с полносвязным слоем по середине и dropuot после слоев нейронов</td>
    <td>10.64%</td>
    <td>Лучший результат</td>
  </tr>
    <tr>
    <td>Test model v2</td>
    <td>С полносвязными слоями с большим числом нейронов ближе к голове</td>
    <td>11.26%</td>
    <td>Результат хуже базовой</td>
  </tr>
  <tr>
    <td>Test model v3</td>
    <td>Мало слоев, без LSTM</td>
    <td>11.58%</td>
    <td>Худший результат</td>
  </tr>
</tbody></table>

**Базовая архитектура показывает лучший результат**

### Tabular + NLP + CV
Создадим модель с подачей на вход табличных данных, текстовых и изображений

Посмотрим на изображения - убедимся, что цены и фото подгрузились верно
<img width="667" alt="img_and_price" src="https://user-images.githubusercontent.com/73405095/198529382-5a81e617-cc73-46fd-ad75-5a6170d6df6e.png">

Изменим размер изображений на (320, 240) и применим аугментацию
<img width="665" alt="image" src="https://user-images.githubusercontent.com/73405095/198529961-3242f8a1-cccf-4405-b4d6-9d8dc15bda72.png">

Подготовим данные таблицы, изображений, текста на вход модели для train, test, sub

#### Построим сеть MLP+NLP+CV
Добавим к лучшей сети MLP+NLP 3-й вход EfficientNetB3 с размороженными слоями
*EfficientNetB3 показала лучший результат среди легковесных SOTA архитектур

    model = Model(inputs=[efficientnet_model.input, model_mlp.input, model_nlp.input], outputs=head)

Обучим сеть, постепенно снижая learning rate и используя EarlyStopping, сохраним веса лучшей сети

    optimizer = tf.keras.optimizers.Adam(ExponentialDecay(1e-3, 100, 0.9))
    earlystop = EarlyStopping(monitor='val_MAPE', patience=10, restore_best_weights=True,)

mape: 11.06%

#### Протестируем сеть MLP+CV
    model = Model(inputs=[efficientnet_model.input, model_mlp.input], outputs=head)
    
mape: 11.16%

### Составлю ансамбли и сделаю предсказание
Для составления предикта присваиваю веса для результата каждой модели ансабля и суммирую их предикты * вес

sub = sum(pred(i) * W(i))

<table>
  <tbody><tr>
    <th>Ансамбль</th>
    <th>Результат (MAPE)</th>
  </tr>
  <tr>
    <td>Only MLP</td>
    <td>11.008</td>
  </tr>
    <tr>
    <td>MLP(0.5) + MLP+NLP(0.3) + MLP+CV(0.2)</td>
    <td>10.989</td>
  </tr>
    <tr>
    <td>MLP(0.4) + MLP+NLP(0.2) + MLP+CV(0.2) + MLP+NLP+CV(0.2)</td>
    <td>11.092</td>
  </tr>
    <tr>
    <td>CatB(0.5) + (MLP(0.5) + MLP+NLP(0.3) + MLP+CV(0.2)) || sum / 2</td>
    <td>10.845</td>
  </tr>
</tbody></table>

Ансамбль CatB(0.5) + (MLP(0.5) + MLP+NLP(0.3) + MLP+CV(0.2)) || sum / 2 - имеет лучший результат
