import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veri=pd.read_csv("odev_tenis.csv")
print(veri)

#VERİ ÖN İŞLEME KISMI
#ilk kolona outlook one hot encoding yapılacak,son iki kolonda label encoding yapılır.
play=veri.iloc[:,-1:].values
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
play[:,-1]=le.fit_transform(veri.iloc[:,-1])
print(play)

#♣tüm kategorikleri numerik value çevirme
from sklearn import preprocessing
veri2=veri.apply(preprocessing.LabelEncoder().fit_transform)
#ilk kolon one_hot_encode
outlook=veri2.iloc[:,:1]
from sklearn import preprocessing
ohe=preprocessing.OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)

weather=pd.DataFrame(data=outlook,index=range(14),columns=['overcast','rainy','sunny'])
sonveri=pd.concat([weather,veri2.iloc[:,3:],veri.iloc[:,1:3]],axis=1)

#regresyon
#humidity bağımlı değişken=y diğerleri=x
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(sonveri.iloc[:,:-1],sonveri.iloc[:,-1:],test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

y_pred=regressor.predict(x_test)
print(y_pred)

#backward elimination en büyük p-value  atmak için
import statsmodels.api as sm
X=np.append(arr=np.ones((14,1)).astype(int),values=sonveri.iloc[:,:-1],axis=1)

X_l=sonveri.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonveri.iloc[:,-1:],X_l).fit()
print(model.summary())

#çıkardığımız değişkenden sonra kalanıyla tekrar modeli eğitip tahmin yaptırdık kurulan 
#modeli y_test e daha yakın sonuç vermesini sağlayıp iyileştirmeye çalışıyoruz.
x_train=x_train.iloc[:,[0,1,2,4,5]]
x_test=x_test.iloc[:,[0,1,2,4,5]]
regressor.fit(x_train, y_train)

y_pred=regressor.predict(x_test)

X_l=sonveri.iloc[:,[0,1,2,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonveri.iloc[:,-1:],X_l).fit()
print(model.summary())
