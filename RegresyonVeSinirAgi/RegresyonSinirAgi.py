import numpy as np
import scipy.io
from numpy import array
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Soru: "Bir yapay zeka modelini kullanarak XOR mantık operasyonunun çözümünü gerçekleştiren bir model tasarlayın
# Modelin eğitim verisi, XOR problemine ait giriş ve hedef değerleri içermektedir.
# Farklı makine öğrenmesi algoritmaları (örneğin, yapay sinir ağları, destek vektör makineleri,
# rastgele orman, XGBoost) kullanarak bu modeli eğitin ve eğitim ile test verileri üzerinde yapılan tahminlerin
# doğruluğunu RMSE (Kök Ortalamalı Kare Hatası) metriği ile değerlendirin. Ayrıca, modelin farklı girişler için
# tahminlerini yapın ve sonuçları görselleştirin."

#CARPMA İSLEMİ MODELİ

#XOR Verisi ve Veri Hazırlığı
xor_input=np.array([[0,0],[0,1],[1,0],[1,1]])
xor_target=np.array([[0],[1],[1],[0]])

data = scipy.io.loadmat('carpma_veriset.mat')
dataset=data['veri']
dataset = dataset.astype('float32')

a2=dataset.shape[1]   #Bu kısımda, veri setinin son sütunu hedef (targ) olarak ayrılır. Geri kalan sütunlar ise öznitelikler (features) olup
# , bu öznitelikler Min-Max ölçekleme yöntemiyle [0,1] aralığına normalize edilir.
targ=dataset[:,a2-1]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset[:,0:a2-1])  # 0 ve 1. stütunları yani sadece öznitelikler

#Veri Bölme
train_size = int(len(dataset) * 0.70) #  %70 anlamında
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
targ_train, targ_test = targ[0:train_size], targ[train_size:len(dataset)]
print("Öznitelikler= ",len(train), len(test),"\n")
print("Sonuçlar= ",len(targ_train), len(targ_test))

trainX=train
trainY=targ_train
testX=test
testY=targ_test

# Modelin Yapılandırılması ve Eğitilmesi

# Yapay Sinir Ağı Modeli (Keras)
model2 = Sequential()
model2.add(Dense(5, input_dim=2, activation='sigmoid'))
model2.add(Dense(5, activation='sigmoid'))
model2.add(Dense(1, activation='linear'))
model2.compile(loss='mean_squared_error', optimizer='adam')
model2.fit(xor_input, xor_target, epochs=1000, verbose=1)
xor_predict = model2.predict(xor_input)
print('xor predicts [0-1-1-0]:',xor_predict)

model = Sequential()
model.add(Dense(40, input_dim=2, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=5000, verbose=1)

model=MLPRegressor((40,20),'relu', max_iter=5000)
model=RandomForestRegressor()
model=SVR(kernel='sigmoid')
model=XGBRegressor()

model.fit(trainX, trainY)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainY = np.reshape(trainY, (trainY.shape[0], 1))
testY = np.reshape(testY, (testY.shape[0], 1))

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
#****************************************************************
#soru sorma
soru=array([[-2,3],[0,0],[120000,120000],[16,3]])
soru = scaler.transform(soru)
soruPredict = model.predict(soru)
print('Sorunun cevabı (42-20-144-48):\n',soruPredict)
#****************************************************************

fige2=plt.figure(figsize=(10,5))
plt.plot(trainY,'bo-',label='orijinal eğitim veri')
plt.plot(trainPredict,'r*-',label='train predict')
plt.legend()
plt.show()

fige3=plt.figure(figsize=(10,5))
plt.plot(testY,'bo-',label='orijinal test veri')
plt.plot(testPredict,'r*-',label='test predict')
plt.legend()
plt.show()