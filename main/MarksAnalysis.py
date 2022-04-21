import pandas as pd
import sqlite3
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

'''address_hf='data/2020.10/google_play_health-fitness.db'
con_hf = sqlite3.connect(address_hf) 
df_hf_19 = pd.read_sql("SELECT * FROM apps", con_hf) 
marks_hf=df_hf_19[{'rating','marks','five','four','three','two','one'}]
marks_hf.rating = pd.to_numeric(marks_hf.rating, errors='coerce')
marks_hf.marks = pd.to_numeric(marks_hf.marks, errors='coerce')
marks_hf.five = [x.replace('%', '') for x in marks_hf.five]
marks_hf.five = pd.to_numeric(marks_hf.five, errors='coerce')
marks_hf.four = [x.replace('%', '') for x in marks_hf.four]
marks_hf.four = pd.to_numeric(marks_hf.four, errors='coerce')
marks_hf.three = [x.replace('%', '') for x in marks_hf.three]
marks_hf.three = pd.to_numeric(marks_hf.three, errors='coerce')
marks_hf.two = [x.replace('%', '') for x in marks_hf.two]
marks_hf.two = pd.to_numeric(marks_hf.two, errors='coerce')
marks_hf.one = [x.replace('%', '') for x in marks_hf.one]
marks_hf.one = pd.to_numeric(marks_hf.one, errors='coerce')

address_m='data/2020.10/google_play_medical.db'
con_m = sqlite3.connect(address_m) 
df_m_19 = pd.read_sql("SELECT * FROM apps", con_m) 
marks_m=df_hf_19[{'rating','marks','five','four','three','two','one'}]
marks_m.rating = pd.to_numeric(marks_m.rating, errors='coerce')
marks_m.marks = pd.to_numeric(marks_m.marks, errors='coerce')
marks_m.five = [x.replace('%', '') for x in marks_m.five]
marks_m.five = pd.to_numeric(marks_m.five, errors='coerce')
marks_m.four = [x.replace('%', '') for x in marks_m.four]
marks_m.four = pd.to_numeric(marks_m.four, errors='coerce')
marks_m.three = [x.replace('%', '') for x in marks_m.three]
marks_m.three = pd.to_numeric(marks_m.three, errors='coerce')
marks_m.two = [x.replace('%', '') for x in marks_m.two]
marks_m.two = pd.to_numeric(marks_m.two, errors='coerce')
marks_m.one = [x.replace('%', '') for x in marks_m.one]
marks_m.one = pd.to_numeric(marks_m.one, errors='coerce')

marks_hf['category'] = 'Health&Care'
marks_m['category'] = 'Medical'
marks= pd.concat([marks_hf, marks_m], ignore_index=True)
marks.to_pickle('marks.pkl')'''

marks = pd.read_pickle('marks.pkl')
rating_mean=marks["rating"].mean()
rating_std=marks["rating"].std()

'''
fig, ax = plt.subplots()
marksnum5_20['rating'].plot(kind = 'density')
marksnum21_100['rating'].plot(kind = 'density')
marksnum101_1000['rating'].plot(kind = 'density')

ax.set(xlabel='average rating per app,-', ylabel='density',
       title='marks distribution')
ax.grid()
ax.legend(title='Number of total ratings given',
          labels=['Apps with 5-20 ratings','Apps with 21-100 ratings','Apps with 101-1000 ratings'])

plt.xlim(1, 5)
plt.ylim(0, 1)
plt.show()'''

marksnum1_50=marks.loc[(marks['marks'] >= 1) & (marks['marks'] <= 50)]
marksnum1_50['range']='1-50'
marksnum51_100=marks.loc[(marks['marks'] >= 51) & (marks['marks'] <= 100)]
marksnum51_100['range']='51-100'
marksnum101_500=marks.loc[(marks['marks'] >= 101) & (marks['marks'] <= 500)]
marksnum101_500['range']='101-500'
marksnum501_1000=marks.loc[(marks['marks'] >= 501) & (marks['marks'] <= 1000)]
marksnum501_1000['range']='501-1000'
marksnum= pd.concat([marksnum1_50, marksnum51_100, marksnum101_500, marksnum501_1000], ignore_index=True)

x=np.arange(0,4,1)
k1_50=marksnum1_50['one'].mean()+marksnum1_50['two'].mean()+marksnum1_50['three'].mean()+marksnum1_50['four'].mean()+marksnum1_50['five'].mean()
k51_100=marksnum51_100['one'].mean()+marksnum51_100['two'].mean()+marksnum51_100['three'].mean()+marksnum51_100['four'].mean()+marksnum51_100['five'].mean()
k101_500=marksnum101_500['one'].mean()+marksnum101_500['two'].mean()+marksnum101_500['three'].mean()+marksnum101_500['four'].mean()+marksnum101_500['five'].mean()
k501_1000=marksnum501_1000['one'].mean()+marksnum501_1000['two'].mean()+marksnum501_1000['three'].mean()+marksnum501_1000['four'].mean()+marksnum501_1000['five'].mean()

yone=[marksnum1_50['one'].mean()/k1_50*100, marksnum51_100['one'].mean()/k51_100*100, marksnum101_500['one'].mean()/k101_500*100, marksnum501_1000['one'].mean()/k501_1000*100]
ytwo=[marksnum1_50['two'].mean()/k1_50*100, marksnum51_100['two'].mean()/k51_100*100, marksnum101_500['two'].mean()/k101_500*100, marksnum501_1000['two'].mean()/k501_1000*100]
ythree=[marksnum1_50['three'].mean()/k1_50*100, marksnum51_100['three'].mean()/k51_100*100, marksnum101_500['three'].mean()/k101_500*100, marksnum501_1000['three'].mean()/k501_1000*100]
yfour=[marksnum1_50['four'].mean()/k1_50*100, marksnum51_100['four'].mean()/k51_100*100, marksnum101_500['four'].mean()/k101_500*100, marksnum501_1000['four'].mean()/k501_1000*100]
yfive=[marksnum1_50['five'].mean()/k1_50*100, marksnum51_100['five'].mean()/k51_100*100, marksnum101_500['five'].mean()/k101_500*100, marksnum501_1000['five'].mean()/k501_1000*100]

fs = 18
width=0.15
labels = ['1-50', '51-100', '101-500', '501-1000']
fig, ax = plt.subplots()
plt.bar(x-2*width,yone,width=width)
plt.bar(x-1*width,ytwo,width=width)
plt.bar(x,ythree,width=width)
plt.bar(x+1*width,yfour,width=width)
plt.bar(x+2*width,yfive,width=width)
plt.xticks(x, labels, fontsize=fs)
plt.yticks(fontsize=fs)

ax.set(xlabel='Количество оценок', ylabel='Процент от всех оценок,%',
       title='-')
ax.xaxis.label.set_size(fs)
ax.yaxis.label.set_size(fs)
ax.grid()
ax.legend(title='Оценки',labels=['1','2','3','4','5'], fontsize=fs)

plt.show()