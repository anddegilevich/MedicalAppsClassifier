import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import collections
from spacy.lang.en.stop_words import STOP_WORDS
import seaborn as sns

address_hf='data/2020.10/google_play_permissions_health-fitness.db'
con_hf = sqlite3.connect(address_hf)
df_hf_19 = pd.read_sql("SELECT * FROM permissions", con_hf)

address_m='data/2020.10/google_play_permissions_medical.db'
con_m = sqlite3.connect(address_m)
df_m_19 = pd.read_sql("SELECT * FROM permissions", con_m)

df_hf_19['category'] = 'Health&Care'
df_hf_19.perm_groups = [x.replace(' ', '_') for x in df_hf_19.perm_groups]
df_m_19['category'] = 'Medical'
df_m_19.perm_groups = [x.replace(' ', '_') for x in df_m_19.perm_groups]
df_19= pd.concat([df_hf_19, df_m_19], ignore_index=True)
N_s=df_19.shape[0]

'''def my_tokenizer(text):
    return text.split() if text != None else []
tokens = df_19.perm_groups.map(my_tokenizer).sum()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS]

token_counter = collections.Counter(remove_stopwords(tokens))

freq_df=pd.DataFrame.from_records(token_counter.most_common(12),
                                  columns=['token', 'count'])

freq_df=freq_df.query("token not in ['Not_Found', 'Other']")

width = 0.5
fig,ax = plt.subplots(1)
freq_df.plot.barh(x='token', y='count', rot=0, color='blue', ax=ax, width=width)

ax.set(xlabel='Most common words', title='-')

plt.show()

topreq=freq_df['token'].to_list()'''

N=10
topreq = ['Photos/Media/Files', 'Storage', 'Location', 'Camera', 'Wi-Fi_connection_information', 'Phone', 'Device_ID_&_call_information', 'Contacts', 'Identity', 'Microphone']

graphdata=np.zeros((N, 3))
for i in range(N):
    mask = df_19["perm_groups"].str.contains(topreq[i])
    res = df_19.loc[mask].shape[0]
    mask_hf = df_hf_19["perm_groups"].str.contains(topreq[i])
    res_hf = df_hf_19.loc[mask_hf].shape[0]
    mask_m = df_m_19["perm_groups"].str.contains(topreq[i])
    res_m = df_m_19.loc[mask_m].shape[0]
    graphdata[i][:]=[res, res_hf, res_m]

x=np.arange(1,11,1)
y=np.transpose(graphdata)
y=y*100/N_s
height = 0.2
fs = 18

plt.subplot(1, 2, 2,)
plt.barh(x-1*height, y[0], height=height)
plt.barh(x, y[1], height=height)
plt.barh(x+1*height, y[2], height=height)

plt.yticks(x, topreq, fontsize=fs-2)
plt.xticks(fontsize=fs)

plt.xlabel('Процент приложений, %', fontsize=fs)
#plt.ylabel('Наиболее частые требования', fontsize=fs)
plt.legend(title='Категория',labels=['Обе категории','Health&Fitness','Medical'], fontsize=fs)
plt.grid()
plt.show()

plt.subplot(1, 1, 1)
CorMatrix=pd.DataFrame()
for i in range(N):
    CorMatrix[topreq[i]]=df_19["perm_groups"].str.contains(topreq[i])

sns.set(font_scale=1.8)
ax = sns.heatmap(CorMatrix.corr(), annot=True, square= True, xticklabels=False, annot_kws={"size": 16})
plt.yticks(fontsize=fs)

plt.show()

