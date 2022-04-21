import pandas as pd
import matplotlib.pyplot as plt
import collections
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from google_trans_new import google_translator

DescriptionS = pd.read_pickle('DescriptionS.pkl')
#ContentS = pd.read_pickle('ContentS.pkl')
df_m_19 = pd.read_pickle('df_m_19.pkl')
translator = google_translator()

DescriptionS['description']=DescriptionS['description'].str.lower()
#df_m_19['description'].str.lower()

def my_tokenizer(text):
    return text.split() if text != None else []
tokens = DescriptionS.description.map(my_tokenizer).sum()
#tokens = DescriptionS.description.sample(100).map(my_tokenizer).sum()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS]
token_counter = collections.Counter(remove_stopwords(tokens))

freq_df=pd.DataFrame.from_records(token_counter.most_common(20),
                                  columns=['token', 'count'])

fs = 18
fig = plt.subplot(2,1,1)
freq_df.plot(kind='bar', x='token', ax=fig)
plt.ylabel('Количество', fontsize=fs)
plt.xlabel('', fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid()
plt.legend().set_visible(False)
plt.show()

def wordcloud(counter):
    wc=WordCloud(width=1200,height=800,
                 background_color="white",
                 max_words=200)
    wc.generate_from_frequencies(counter)
    fig=plt.figure(figsize=(6,4))
    plt.imshow(wc,interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

wordcloud(token_counter)

'''mask = df_m_19["description"].str.contains('health|medical|weight|body|fitness|training|patient')
res = df_m_19.loc[mask]

x=df_m_19['size']
y=df_m_19['installs']
xT=res['size']
yT=res['installs']

fig, ax = plt.subplots()

ax.scatter(x, y,
           c = 'black')
ax.scatter(xT, yT, c='red')

ax.set_facecolor('white')
ax.set_title('Скаттерграмма')
ax.legend(['Иначальные данные','После фильтрации по тексту'])

plt.xlabel("Size")
plt.ylabel("Installs")
plt.xlim(0, 500)
plt.ylim(0, 101000)

plt.show()


#top=res.nlargest(10, ['installs'])[['name','installs']]'''


