import pandas as pd
import sqlite3
import os

files = os.listdir(path="./data")
N=len(files)

RatingS_hf=pd.DataFrame({'Rating': [], 'Month':[]})
RatingS_m=pd.DataFrame({'Rating': [], 'Month':[]})
RatingS=pd.DataFrame({'Rating': [], 'Month':[]})
SizeS_hf=pd.DataFrame({'Size': [], 'Month':[]})
SizeS_m=pd.DataFrame({'Size': [], 'Month':[]})
SizeS=pd.DataFrame({'Size': [], 'Month':[]})
AndroidS_hf=pd.DataFrame({'Android': [], 'Month':[]})
AndroidS_m=pd.DataFrame({'Android': [], 'Month':[]})
AndroidS=pd.DataFrame({'Android': [], 'Month':[]})

for i in range(N):
    address_hf='data/'+files[i]+'/google_play_health-fitness.db'
    con_hf = sqlite3.connect(address_hf)  # Путь к базе данных
    globals()['df_hf_%s' % i] = pd.read_sql("SELECT * FROM apps", con_hf) #Считывание

    mask_hf = globals()['df_hf_%s' % i].description.str.contains(
        'Health|health|Medical|medical|weight|body|fitness|training|patient')
    globals()['df_hf_%s' % i] = globals()['df_hf_%s' % i].loc[mask_hf]

    globals()['df_hf_%s' % i].rating = pd.to_numeric(globals()['df_hf_%s' % i].rating, errors='coerce')  # Преобразывание типа данных в числовой
    Rating_hf = pd.DataFrame({'Rating': globals()['df_hf_%s' % i].rating, 'Month':files[i]+'.'})
    RatingS_hf=pd.concat([RatingS_hf, Rating_hf], ignore_index=True)

    globals()['df_hf_%s' % i].installs = [x.replace(',', '') for x in globals()['df_hf_%s' % i].installs]
    globals()['df_hf_%s' % i].installs = [x.replace('+', '') for x in globals()['df_hf_%s' % i].installs]
    globals()['df_hf_%s' % i].installs = pd.to_numeric(globals()['df_hf_%s' % i].installs, errors='coerce')

    globals()['df_hf_%s' % i]['size'] = [x.replace('M', '') for x in globals()['df_hf_%s' % i]['size']]
    globals()['df_hf_%s' % i]['size'] = pd.to_numeric(globals()['df_hf_%s' % i]['size'], errors='coerce')
    Size_hf = pd.DataFrame({'Size': globals()['df_hf_%s' % i]['size'], 'Month': files[i] + '.'})
    SizeS_hf = pd.concat([SizeS_hf, Size_hf], ignore_index=True)

    globals()['df_hf_%s' % i].android = [x.replace(' and up', '') for x in globals()['df_hf_%s' % i].android]
    globals()['df_hf_%s' % i].android = pd.to_numeric(globals()['df_hf_%s' % i].android, errors='coerce')

    Android_hf = pd.DataFrame({'Android': globals()['df_hf_%s' % i].android, 'Month': files[i] + '.'})
    AndroidS_hf = pd.concat([AndroidS_hf, Android_hf], ignore_index=True)

    globals()['df_hf_%s' % i].rating = pd.to_numeric( globals()['df_hf_%s' % i].rating, errors='coerce')
    globals()['df_hf_%s' % i].marks = pd.to_numeric( globals()['df_hf_%s' % i].marks, errors='coerce')
    globals()['df_hf_%s' % i].five = [x.replace('%', '') for x in  globals()['df_hf_%s' % i].five]
    globals()['df_hf_%s' % i].five = pd.to_numeric( globals()['df_hf_%s' % i].five, errors='coerce')
    globals()['df_hf_%s' % i].four = [x.replace('%', '') for x in  globals()['df_hf_%s' % i].four]
    globals()['df_hf_%s' % i].four = pd.to_numeric( globals()['df_hf_%s' % i].four, errors='coerce')
    globals()['df_hf_%s' % i].three = [x.replace('%', '') for x in  globals()['df_hf_%s' % i].three]
    globals()['df_hf_%s' % i].three = pd.to_numeric( globals()['df_hf_%s' % i].three, errors='coerce')
    globals()['df_hf_%s' % i].two = [x.replace('%', '') for x in  globals()['df_hf_%s' % i].two]
    globals()['df_hf_%s' % i].two = pd.to_numeric( globals()['df_hf_%s' % i].two, errors='coerce')
    globals()['df_hf_%s' % i].one = [x.replace('%', '') for x in  globals()['df_hf_%s' % i].one]
    globals()['df_hf_%s' % i].one = pd.to_numeric( globals()['df_hf_%s' % i].one, errors='coerce')

    #globals()['df_hf_%s' % i]['timestamp'] = globals()['df_hf_%s' % i]['timestamp'].apply(pd.to_datetime)
    #-------------------------------------------------------------------------------------------------------------------
    address_m = 'data/' + files[i] + '/google_play_medical.db'
    con_m = sqlite3.connect(address_m)
    globals()['df_m_%s' % i] = pd.read_sql("SELECT * FROM apps", con_m)

    mask_m = globals()['df_m_%s' % i].description.str.contains(
        'Health|health|Medical|medical|weight|body|fitness|training|patient')
    globals()['df_m_%s' % i] = globals()['df_m_%s' % i].loc[mask_m]

    globals()['df_m_%s' % i].rating = pd.to_numeric(globals()['df_m_%s' % i].rating,errors='coerce')
    Rating_m = pd.DataFrame({'Rating': globals()['df_m_%s' % i].rating, 'Month': files[i] + '.'})
    RatingS_m = pd.concat([RatingS_m, Rating_m], ignore_index=True)

    globals()['df_m_%s' % i].installs = [x.replace(',', '') for x in globals()['df_m_%s' % i].installs]
    globals()['df_m_%s' % i].installs = [x.replace('+', '') for x in globals()['df_m_%s' % i].installs]
    globals()['df_m_%s' % i].installs = pd.to_numeric(globals()['df_m_%s' % i].installs, errors='coerce')

    globals()['df_m_%s' % i]['size'] = [x.replace('M', '') for x in globals()['df_m_%s' % i]['size']]
    globals()['df_m_%s' % i]['size'] = pd.to_numeric(globals()['df_m_%s' % i]['size'], errors='coerce')
    Size_m = pd.DataFrame({'Size': globals()['df_m_%s' % i]['size'], 'Month': files[i] + '.'})
    SizeS_m = pd.concat([SizeS_m, Size_m], ignore_index=True)

    globals()['df_m_%s' % i].android = [x.replace(' and up', '') for x in globals()['df_m_%s' % i].android]
    globals()['df_m_%s' % i].android = pd.to_numeric(globals()['df_m_%s' % i].android, errors='coerce')

    Android_m = pd.DataFrame({'Android': globals()['df_m_%s' % i].android, 'Month': files[i] + '.'})
    AndroidS_m = pd.concat([AndroidS_m, Android_m], ignore_index=True)

    globals()['df_m_%s' % i].rating = pd.to_numeric(globals()['df_m_%s' % i].rating, errors='coerce')
    globals()['df_m_%s' % i].marks = pd.to_numeric(globals()['df_m_%s' % i].marks, errors='coerce')
    globals()['df_m_%s' % i].five = [x.replace('%', '') for x in globals()['df_m_%s' % i].five]
    globals()['df_m_%s' % i].five = pd.to_numeric(globals()['df_m_%s' % i].five, errors='coerce')
    globals()['df_m_%s' % i].four = [x.replace('%', '') for x in globals()['df_m_%s' % i].four]
    globals()['df_m_%s' % i].four = pd.to_numeric(globals()['df_m_%s' % i].four, errors='coerce')
    globals()['df_m_%s' % i].three = [x.replace('%', '') for x in globals()['df_m_%s' % i].three]
    globals()['df_m_%s' % i].three = pd.to_numeric(globals()['df_m_%s' % i].three, errors='coerce')
    globals()['df_m_%s' % i].two = [x.replace('%', '') for x in globals()['df_m_%s' % i].two]
    globals()['df_m_%s' % i].two = pd.to_numeric(globals()['df_m_%s' % i].two, errors='coerce')
    globals()['df_m_%s' % i].one = [x.replace('%', '') for x in globals()['df_m_%s' % i].one]
    globals()['df_m_%s' % i].one = pd.to_numeric(globals()['df_m_%s' % i].one, errors='coerce')

    #globals()['df_m_%s' % i]['timestamp'] = globals()['df_m_%s' % i]['timestamp'].apply(pd.to_datetime)
    print(i)

RatingS_hf['Category'] = 'Health&Care'
RatingS_m['Category'] = 'Medical'
RatingS= pd.concat([RatingS_hf, RatingS_m], ignore_index=True)
RatingS.to_pickle('RatingS.pkl')

SizeS_hf['Category'] = 'Health&Care'
SizeS_m['Category'] = 'Medical'
SizeS = pd.concat([SizeS_hf, SizeS_m], ignore_index=True)
SizeS.to_pickle('SizeS.pkl')

AndroidS_hf['Category'] = 'Health&Care'
AndroidS_m['Category'] = 'Medical'
AndroidS= pd.concat([AndroidS_hf, AndroidS_m], ignore_index=True)
AndroidS.to_pickle('AndroidS.pkl')

names_top10_hf = df_hf_22.nlargest(10, ['installs']).name
names_top10_m =  df_m_22.nlargest(10, ['installs']).name
df_top10S_hf = pd.DataFrame({'Name': [],'Rating': [], 'Month':[]})
df_top10S_m = pd.DataFrame({'Name': [],'Rating': [], 'Month':[]})

for i in range(N):
    df_top10_hf = globals()['df_hf_%s' % i].query("name in @names_top10_hf")
    df_top10_hf =  pd.DataFrame({'Name': df_top10_hf['name'], 'Rating': df_top10_hf['rating'], 'Month': files[i]+'.'})
    df_top10S_hf = pd.concat([df_top10S_hf, df_top10_hf], ignore_index=True)

    df_top10_m = globals()['df_m_%s' % i].query("name in @names_top10_m")
    df_top10_m = pd.DataFrame({'Name': df_top10_m['name'], 'Rating': df_top10_m['rating'], 'Month': files[i] + '.'})
    df_top10S_m = pd.concat([df_top10S_m, df_top10_m], ignore_index=True)

#df_top10S_hf.to_pickle('df_top10S_hf.pkl')
#df_top10S_m.to_pickle('df_top10S_m.pkl')

df_top10S_hf['Category'] = 'Health&Care'
df_top10S_m['Category'] = 'Medical'
df_top10S= pd.concat([df_top10S_hf, df_top10S_m], ignore_index=True)
df_top10S.to_pickle('df_top10S.pkl')

Description_hf = pd.DataFrame(df_hf_22.description)
Description_m = pd.DataFrame(df_m_22.description)
Description_hf['Category'] = 'Health&Care'
Description_m['Category'] = 'Medical'
DescriptionS= pd.concat([Description_hf, Description_m], ignore_index=True)
DescriptionS.to_pickle('DescriptionS.pkl')

Content_hf = pd.DataFrame(df_hf_22.content)
Content_m = pd.DataFrame(df_m_22.content)
Content_hf['Category'] = 'Health&Care'
Content_m['Category'] = 'Medical'
ContentS= pd.concat([Content_hf, Content_m], ignore_index=True)
ContentS.to_pickle('ContentS.pkl')

for i in range(N):
    globals()['df_hf_%s' % i]['category'] = 'Health&Care'
    globals()['df_m_%s' % i]['category'] = 'Medical'
    globals()['df_%s' % i] = pd.concat([globals()['df_hf_%s' % i],globals()['df_m_%s' % i]], ignore_index=True)
    globals()['df_%s' % i].to_pickle('dadata/df_%s.pkl' % i)