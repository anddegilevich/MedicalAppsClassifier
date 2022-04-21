import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

files = os.listdir(path="./data")

Types = ["Новости", "Информация", "Обучающий материал", "Проигрыватель", "Брокер",
                      "Поддержка принятия решений", "Калькулятор", "Измерительный прибор", "Монитор", "Трекер",
                      "Административные задачи", "Дневник", "Напоминание", "Календарь", "Помощь", "Тренер",
                      "Менеджер по здоровью", "Привод", "Коммуникатор", "Игра", "Магазин", "Прочее"]
'''Month = []
for i in range(23):
    Predictions = np.loadtxt('dadata/df_%s_Type.txt' % i)
    pred_df = pd.DataFrame({'label': Predictions})
    y = pred_df['label'].value_counts().sort_index(ascending=True)
    #y = y/np.sum(y)*100
    if i==0:
        y_sum = y
        N = len(Predictions)
    else:
        y_sum = np.vstack([y_sum, y])
        N = np.vstack([N, len(Predictions)])
    Month = np.hstack([Month, files[i]])
    print(i)


number_of_colors = 22
color = ["%06x" % np.random.randint(0, 0xFFFFFF) for i in range(number_of_colors)]'''

'''plt.stackplot(Month,y_sum.transpose(), colors=color)
plt.xticks(rotation=45)
plt.legend(Types)
plt.ylabel('Процент рынка')
plt.show()'''

'''plt.plot(N)
plt.xticks(rotation=45)
plt.legend(Types)
plt.ylabel('Количество приложений')
plt.show()'''

Types = ["Новости", "Информация", "Обучающий материал", "Проигрыватель", "Брокер",
                      "Поддержка принятия решений", "Калькулятор", "Измерительный прибор", "Монитор", "Трекер",
                      "Административные задачи", "Дневник", "Напоминание", "Календарь", "Помощь", "Тренер",
                      "Менеджер по здоровью", "Привод", "Коммуникатор", "Игра", "Магазин", "Прочее"]

Disease = ["Ожирение", "Женское здоровье", "Мышечная слабость", "Болезнь сердца",
                             "Психологическое заболевание", "Онкологическое заболевание", "Болезнь ротовой полости",
                             "Болезнь костей и суставов", "Диабет", "Болезнь Печени", "Комплексное заболевание",
                             "Болезнь животных", "Другое"]

N=len(Disease)
Predictions = np.loadtxt('dadata/df_22_Disease.txt')
pred_df = pd.DataFrame({'label': Predictions})
y = pred_df['label'].value_counts().sort_index(ascending=True)
y_sum = sum(y)
print(y)
x = np.arange(N)+1
width=0.5
fs=18
plt.subplot(1, 2, 2)
plt.barh(x, y*100/y_sum, height=width, color='blue')
plt.xlabel('Процент, %', fontsize=fs)
plt.ylabel('Номер класса', fontsize=fs)
plt.yticks(x, labels=Disease, fontsize=fs)
plt.grid()
plt.show()