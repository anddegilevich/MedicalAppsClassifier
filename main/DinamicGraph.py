import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RatingS = pd.read_pickle('RatingS.pkl')
SizeS = pd.read_pickle('SizeS.pkl')

fs = 18
'''sns.boxplot(y='Month', x='Rating', hue="Category",
                 data=RatingS, orient="h")
plt.ylabel('Месяц', fontsize=fs)
plt.xlabel('Рэйтинг', fontsize=fs)
#plt.legend(title='Категория',labels=['Health&Fitness','Medical'], fontsize=fs)
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,fontsize=fs-8)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid()
plt.show()'''

sns.boxplot(y='Month', x='Size', hue="Category",
                 data=SizeS, orient="h")
plt.ylabel('Месяц', fontsize=fs)
plt.xlabel('Размер, Мбайт', fontsize=fs)
#plt.legend(title='Категория',labels=['Health&Fitness','Medical'], fontsize=fs)
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,fontsize=fs-8)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid()
plt.show()