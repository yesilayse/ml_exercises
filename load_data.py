x=10
class insan:
    boy=180
    def kosmak(self,b):
        return b+10
ali=insan()
print(ali.boy)
print(ali.kosmak(90))
l=[1,3,4]#liste

# kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri önişleme
#load data
veriler=pd.read_csv('eksikveriler.csv')
print(veriler)
boy=veriler[['boy']]
print(boy)
boykilo=veriler[['boy','kilo']]
print(boykilo)

#eksikveriler
#sci-kit learn
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
yas=veriler.iloc[:,1:4].values
print(yas)
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)

#encoder nominal,ordinal verilerin numeric veriye dönüştürülmesi
#kategorik verilerin dönüştürülmesi
ülke=veriler.iloc[:,0:1].values
print(ülke)
from sklearn import preprocessing
#label encoder her farklı değere 0 dan başlayarak değer verir
le=preprocessing.LabelEncoder()
ülke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ülke)

#one hotencoder kolon başlıklarına etiketleri taşımak ve her etiketin altına 1-0 atar
ohe=preprocessing.OneHotEncoder()
ülke=ohe.fit_transform(ülke).toarray()
print(ülke)

#numpy dizilerin dataframe dönüşümü ve dataframe birleştirilmesi
#verilerin birleştirilmesi,dataframelerin birleştirilmes
sonuc=pd.DataFrame(data=ülke,index=range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data=yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)
sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#verilerin train ve test olarak bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)

#öznitelik ölçeklendirme,verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
























