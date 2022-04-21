import pandas as pd
import tensorflow as tf
from keras import models
from keras import layers
from keras import backend as K
from keras import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sqlite3
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow.keras.datasets import mnist

'''data = pd.read_pickle('df_22.pkl')
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
data=data.dropna()
data=data.drop(['app_id', 'new', 'updated', 'version', 'in_app_products', 'developer',
           'developer_website','developer_mail','developer_privacy','developer_address',
           'timestamp'], axis='columns')'''


#data.sample(1000).to_excel('data.xlsx')

data=pd.read_excel('dataNN.xlsx', index_col=0)
data=data.head(300)

Types = ["Новости", "Информация", "Обучающий материал", "Проигрыватель", "Брокер",
                      "Поддержка принятия решений", "Калькулятор", "Измерительный прибор", "Монитор", "Трекер",
                      "Административные задачи", "Дневник", "Напоминание", "Календарь", "Помощь", "Тренер",
                      "Менеджер по здоровью", "Привод", "Коммуникатор", "Игра", "Магазин", "Прочее"]
Diseases = ["Диета и потеря веса", "Беременность и менструальный цикл", "Психология", "Спорт и фитнесс",
                        "Кардио", "Рак", "Стоматология", "Комплексная терапия", "Другое", "Кости и суставы",
                        "Ветеринария", "Диабет", "Печень"]
for j in range(len(Types)):
 data['Type'] = data['Type'].replace([j+1], Types[j])
for j in range(len(Diseases)):
 data['Disease'] = data['Disease'].replace([j+1], Diseases[j])

conn = sqlite3.connect("data.db")
data.to_sql('data', con=conn, if_exists='append')
quit()

data=data.drop(columns=['name', 'description', 'Disease'])
categorical_parametrs=['content', 'category', 'interactive_elements']
for i in range(len(categorical_parametrs)):
 data[categorical_parametrs[i]] = pd.Categorical(data[categorical_parametrs[i]])
 data[categorical_parametrs[i]] = data[categorical_parametrs[i]].cat.codes

target = data.pop('Type')
x, x_test, y, y_test = train_test_split(data,target,test_size=0.2,train_size=0.8)

training_df = MinMaxScaler().fit_transform(x)
testing_df = MinMaxScaler().fit_transform(x_test)

train_x=tf.convert_to_tensor(training_df)
train_y=tf.convert_to_tensor(y)

test_x=tf.convert_to_tensor(testing_df)
test_y=tf.convert_to_tensor(y_test)

def coeff_determination(y_true, y_pred):
 SS_res = K.sum(K.square(y_true - y_pred))
 SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
 return (1 - SS_res / (SS_tot + K.epsilon()))

model = models.Sequential()
model.add(layers.Dense(26, activation = "relu", input_shape=(data.shape[1],)))
model.add(layers.Dense(52, activation = "relu"))
model.add(layers.Dense(104, activation = "relu"))
model.add(layers.Dense(52, activation = "relu"))
model.add(layers.Dense(1,activation="linear"))
model.summary()

model.compile(
 optimizer = 'adam',
 loss = 'categorical_crossentropy',
 #metrics = [coeff_determination]
 metrics = ['accuracy']
)

results = model.fit(
 train_x, train_y,
 epochs = 1000,
 batch_size = 100,
 validation_data=(test_x, test_y)
)

print(results.history.keys())
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
