import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from english_words import english_words_set
from google_trans_new import google_translator
import nltk
import torch
import gensim.downloader as api

df = pd.read_pickle('df_22.pkl')
Ndf = df.shape[0]
description = df['description'].head(Ndf)

tokenizer = nltk.WordPunctTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

description = description.apply(tokenizer.tokenize)

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]
description = description.apply(lemmatize_text)

word2vec = api.load('glove-wiki-gigaword-300')
word2idx = {word: idx for idx, word in enumerate(word2vec.index2word)}

def encode(word):
    if word in word2idx.keys():
        return word2idx[word]
    return word2idx['unk']

def encode_text(text):
    return [encode(w) for w in text]
description = description.apply(encode_text)

for i in range(Ndf):
    description.loc[i] = np.fromiter(description[i], dtype=np.int)

quit()

data = pd.read_excel('dataNN.xlsx', index_col=0)
N_apps = 600
N_test = 50
N_train = int(N_apps - N_test)

data = data.head(N_apps)

data['Type'] = data['Type'] - 1

'''data['Type'] = np.where((data['Type'] >= 0) & (data['Type'] <= 4), int(0), data['Type'])
data['Type'] = np.where((data['Type'] >= 5) & (data['Type'] <= 9), int(1), data['Type'])
data['Type'] = np.where((data['Type'] >= 10) & (data['Type'] <= 10), int(2), data['Type'])
data['Type'] = np.where((data['Type'] >= 11) & (data['Type'] <= 13), int(3), data['Type'])
data['Type'] = np.where((data['Type'] >= 14) & (data['Type'] <= 16), int(4), data['Type'])
data['Type'] = np.where((data['Type'] >= 17) & (data['Type'] <= 21), int(5), data['Type'])'''

data['Type'] = data['Type'].astype('int')

data['Disease']= data['Disease'] - 1
data['Disease'] = data['Disease'].astype('int')

data = data.sample(frac=1, replace=True, random_state=9)

Types = ["Новости", "Информация", "Обучающий материал", "Проигрыватель", "Брокер",
                      "Поддержка принятия решений", "Калькулятор", "Измерительный прибор", "Монитор", "Трекер",
                      "Административные задачи", "Дневник", "Напоминание", "Календарь", "Помощь", "Тренер",
                      "Менеджер по здоровью", "Привод", "Коммуникатор", "Игра", "Магазин", "Прочее"]

Disease_text = ["Ожирение", "Женское здоровье", "Мышечная слабость", "Болезнь сердца",
                             "Психологическое заболевание", "Онкологическое заболевание", "Болезнь ротовой полости",
                             "Болезнь костей и суставов", "Диабет", "Болезнь Печени", "Комплексное заболевание",
                             "Болезнь животных", "Другое"]

Diseases = ["Ожирение", "Женское здоровье", "Психологическое заболевание", "Мышечная слабость",
                        "Болезнь сердца", "Онкологическое заболевание", "Болезнь ротовой полости",
                        "Комплексное заболевание", "Другое", "Болезнь костей и суставов",
                        "Болезнь животных", "Диабет", "Болезнь Печени"]

categories = ['Предоставление информации', 'Приложения для обработки данных', 'Приложения для управления',
            'Календари и дневники', 'Приложения поддержки', 'Прочие']

new_num = [1, 2, 5, 3, 4, 6, 7, 11, 13, 8, 12, 9, 10 ]

for j in range(len(Diseases)):
    data['Disease'] = data['Disease'].replace([j], Diseases[j])

for j in range(len(Diseases)):
    data['Disease'] = data['Disease'].replace(Diseases[j], new_num[j])

'''for j in range(len(Types)):
 data['Type'] = data['Type'].replace([j], Types[j])'''

#data = data.reset_index(drop=True)

'''conn = sqlite3.connect("data.db")
data.to_sql('data', con=conn, if_exists='append')'''

#feature='Type'
feature='Disease'

train_df = pd.DataFrame({'text': data['description'].head(N_train), 'label': data[feature].head(N_train)})
test_df = pd.DataFrame({'text': data['description'].tail(N_test), 'label': data[feature].tail(N_test)})

a = data['description'][0]

if english_words_set not in a:
     a.replace(a, "")

#Рисунок 8
'''plt.subplot(1, 2, 2)
fs = 18
N = train_df['label'].nunique()
#x = np.arange(N)+1
x = Types
y = data['Type'].value_counts().sort_index(ascending=True)
xG1 = ["Информация", "Брокер", "Новости", "Обучающий материал", "Проигрыватель"]
yG1 = [y[1], y[4], y[0], y[2], y[3]]
xG2 = ["Монитор", "Измерительный прибор", "Калькулятор", "Поддержка принятия решений", "Трекер"]
yG2 = [y[8], y[7], y[6], y[5], y[9]]
xG3 = ["Административные задачи"]
yG3 = [y[10]]
xG4 = ["Дневник", "Календарь", "Напоминание"]
yG4 = [y[11], y[13], y[12]]
xG5 = ["Тренер", "Менеджер по здоровью", "Помощь"]
yG5 = [y[15], y[16], y[14]]
xG6 = ["Коммуникатор", "Магазин", "Прочее", "Игра", "Привод"]
yG6 = [y[18], y[20], y[21], y[19], y[17]]

print(sum(yG6))'''
'''plt.barh(x[0:5], y[0:5])
plt.barh(x[5:10], y[5:10])
plt.barh(x[10:11], y[10:11])
plt.barh(x[11:14], y[11:14])
plt.barh(x[14:17], y[14:17])
plt.barh(x[17:22], y[17:22])'''

'''plt1 = plt.barh(xG6, yG6)
plt2 = plt.barh(xG5, yG5)
plt3 = plt.barh(xG4, yG4)
plt4 = plt.barh(xG3, yG3)
plt5 = plt.barh(xG2, yG2)
plt6 = plt.barh(xG1, yG1)
plt.xlabel('Количество', fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(np.arange(0, 100, 10), fontsize=fs)
leg = ['Предоставление информации', 'Приложения для обработки данных', 'Приложения для управления',
            'Календари и дневники', 'Приложения поддержки', 'Прочие']
leg2 = ['Прочие', 'Приложения поддержки', 'Календари и дневники', 'Приложения для управления',
        'Приложения для обработки данных', 'Предоставление информации']
#plt.legend(leg2, fontsize=fs-6)
plt.grid()
plt.show()'''

#Рисунок 18-20
'''width = 0.4
fs = 18
N = train_df['label'].nunique()
x = np.arange(N)+1
ytrain = train_df['label'].value_counts().sort_index(ascending=True)
ytest = test_df['label'].value_counts().sort_index(ascending=True)'''

'''ytest[5] = 0
ytest[6] = 0
ytest[7] = 0
ytest[9] = 0
ytest[12] = 0
ytest[14] = 0
ytest[17] = 0
ytest[19] = 0
ytest[21] = 0'''

'''ytest[4] = 0
ytest[6] = 0
ytest[7] = 0
ytest[10] = 0

ytest = ytest.sort_index(ascending=True)
print(ytest)'''

'''plt.subplot(1, 2, 1)
plt.bar(x , ytrain, width=width, color='blue')
plt.ylabel('Количество', fontsize=fs)
plt.xlabel('Номер класса', fontsize=fs)
plt.xticks(x, fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(title='Выборка',labels=['Обучающая'], fontsize=fs)
plt.grid()

plt.subplot(1, 2, 2)
plt.bar(x , ytest, width=width, color='red')
plt.ylabel('Количество', fontsize=fs)
plt.xlabel('Номер класса', fontsize=fs)
plt.xticks(x, fontsize=fs)
plt.yticks(np.arange(0, max(ytest)+1, 2.0),fontsize=fs)
plt.legend(title='Выборка',labels=['Тестовая'], fontsize=fs)
plt.grid()'''

'''plt.subplot(1, 2, 2)
plt.barh(x-0.5*width, ytrain, height=width, color='blue')
plt.barh(x+0.5*width, ytest, height=width, color='red')
plt.xlabel('Количество', fontsize=fs)
plt.ylabel('Номер класса', fontsize=fs)
plt.yticks(x, labels=Disease_text, fontsize=fs)
#plt.xticks(np.arange(0, max(ytest)+1, 2.0),fontsize=fs)
plt.legend(title='Выборка',labels=['Обучающая', 'Тестовая'], fontsize=fs)
plt.grid()
plt.show()'''

#Рисунок 21
'''fs = 18
x = np.arange(200)+1
y_test = np.loadtxt('CNN_Test_Disease_LR-3.txt')
y_train = np.loadtxt('CNN_Train_Disease_LR-3.txt') - 0.24

plt.subplot(1, 2, 1)
plt.plot(x[4:200] ,y_train[1][4:200])
plt.plot(x[4:200] ,y_train[2][4:200])
plt.title('Обучающая выборка', fontsize=fs)
plt.ylabel('Значение статистического критерия', fontsize=fs)
plt.xlabel('Номер эпохи обучения', fontsize=fs)
plt.legend(title='Критерий',labels=['Точность','F-мера'], fontsize=fs)
plt.xlim(1, 200)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylim(0,0.7)
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x ,y_test[1])
plt.plot(x ,y_test[2])
plt.title('Тестовая выборка', fontsize=fs)
plt.ylabel('Значение статистического критерия', fontsize=fs)
plt.xlabel('Номер эпохи обучения', fontsize=fs)
plt.legend(title='Критерий',labels=['Точность','F-мера'], fontsize=fs)
plt.xlim(1, 200)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid()
plt.show()

print(max(y_train[1]))'''
