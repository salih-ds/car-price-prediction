import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
import pymorphy2
from string import punctuation
from tensorflow.keras.preprocessing import sequence


# создать датафрейм для единого преобразования текстов
def create_nlp_df(train, test):
	# train - необработанные тренировочные данные
	# test - необработанные тестовые данные

	data_nlp = test.append(train, sort=False).reset_index(drop=True)
	data_nlp = data_nlp[['description', 'sample', 'price']]

	return(data_nlp)


# перевести тексты в нижний регистр, очистить от символов вне паттерна, применить лемматизацию, удалить стоп-слова
def clear_nlp_data(data_nlp, patterns, morph, russian_stopwords):
	# data_nlp - объединенный датафрейм со всеми текстами для преобразования
	# patterns - определить, какие символы оставить в тексте
	# morph - функция лемматизации
	# russian_stopwords - список стоп-слов

	# переведем в нижний регист
	data_nlp['description'] = data_nlp['description'].str.lower()

	def clear_str(data_str):
	    data_str = data_str.replace('Ё', 'Е').replace('ё', 'е')
	    data_str = re.sub(patterns, ' ', data_str)
	    return data_str

	data_nlp['description'] = data_nlp.apply(
	    lambda data_nlp: clear_str(data_nlp.description), axis=1)

	def lemmatize_stopword(text):
	    tokens = []
	    for token in text.split():
	        if token not in russian_stopwords and token.strip() not in punctuation:
	          token = token.strip()
	          token = morph.normal_forms(token)[0]
	          tokens.append(token)
	    return ' '.join(tokens)

	data_nlp['description'] = data_nlp.apply(
	    lambda data_nlp: lemmatize_stopword(data_nlp.description), axis=1)

	return(data_nlp)


# Токенизация "Мешок слов"
def bag_of_words_tokenize(text_train, text_test, text_sub, all_texts, MAX_WORDS=100000, MAX_SEQUENCE_LENGTH=256):
	# text_train - тексты из тренировочного массива
	# text_test - текст из валидационного массива
	# text_sub - текст для предикта на соревнование
	# all_texts - все тексты для составления словаря
	# MAX_WORDS - максимальное число слов в словаре
	# MAX_SEQUENCE_LENGTH - максимальная длинна вектора

	# составим словарь
	tokenize = Tokenizer(num_words=MAX_WORDS)
	tokenize.fit_on_texts(all_texts)

	# векторизуем текст
	text_train_sequences = sequence.pad_sequences(
	    tokenize.texts_to_sequences(text_train), maxlen=MAX_SEQUENCE_LENGTH)
	text_test_sequences = sequence.pad_sequences(
	    tokenize.texts_to_sequences(text_test), maxlen=MAX_SEQUENCE_LENGTH)
	text_sub_sequences = sequence.pad_sequences(
	    tokenize.texts_to_sequences(text_sub), maxlen=MAX_SEQUENCE_LENGTH)

	# индексы слов
	word_index = tokenize.word_index
	
	return(text_train_sequences, text_test_sequences, text_sub_sequences, tokenize)


# Токенизация по словосочетанию из 2-х слов
def two_gram_tokenize(X_train_index, X_test_index, X_sub_index, all_texts, MAX_PAIRS = 256):
	# X_train_index, X_test_index, X_sub_index - индексы из общего датафрейма
	# all_texts - все тексты для составления словаря
	# MAX_PAIRS - количество пар токенов в векторе

	# составим словарь и векторизуем текстовые данные
	cv = CountVectorizer(ngram_range=(2,2), max_features=MAX_PAIRS)
	count_vector=cv.fit_transform(all_texts)

	# приведем данные в формат для входа модели
	text_2 = pd.DataFrame(count_vector.toarray().transpose(),index=cv.get_feature_names()).transpose()

	# разделим данные
	text_train_sequences_2 = text_2.iloc[X_train_index]
	text_test_sequences_2 = text_2.iloc[X_test_index]
	text_sub_sequences_2 = text_2.iloc[X_sub_index]

	return(text_train_sequences_2, text_test_sequences_2, text_sub_sequences_2)






