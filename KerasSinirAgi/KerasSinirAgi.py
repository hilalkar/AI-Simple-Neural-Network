import scipy.io
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras.optimizers.legacy import SGD

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# kfold = KFold(10, shuffle=True, random_state=42)
# for train, test in kfold.split(target):


veri = scipy.io.loadmat('iris_deneme1.mat')
train_inp=veri['train_inp']
train_tar=veri['train_tar']
test_inp=veri['test_inp']
test_tar=veri['test_tar']

train_inp = train_inp.astype('float32')
test_inp = test_inp.astype('float32')

# one hot encode outputs
train_tar = np_utils.to_categorical(train_tar)
test_tar = np_utils.to_categorical(test_tar)
num_classes = 3

model = Sequential()
model.add(Dense(20, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])  # sparse_categorical_crossentropy   or   categorical_crossentropy

# Fit the model
model.fit(train_inp, train_tar, epochs=100, batch_size=8)
print(model.summary())
# Final evaluation of the model
_, train_acc = model.evaluate(train_inp, train_tar, verbose=0)
_, test_acc = model.evaluate(test_inp, test_tar, verbose=0)
print("Train Accuracy: %.2f%%" % (train_acc * 100))
print("Test Accuracy: %.2f%%" % (test_acc * 100))



# "Bir yapay sinir ağı kullanarak, Iris veri setindeki çiçek türlerini sınıflandıracak bir model tasarlayın ve bu modeli Keras kütüphanesiyle Python’da geliştirin.
# Modelin eğitim sürecini aşağıdaki adımlarla tamamlayın ve sonuçları değerlendirin:**
#
# Eğitim ve test verilerini MATLAB formatında (örneğin, iris_deneme1.mat) verilen dosyadan yükleyin. Bu dosya train_inp, train_tar, test_inp, test_tar adlı dört matris içermektedir.
# Verileri float32 veri tipine dönüştürün ve sınıf hedeflerini One-Hot Encoding yöntemi ile kodlayın.
# Keras Sequential API kullanarak, 20 nöronlu relu aktivasyonlu bir katman ve ardından sınıf sayısı kadar nöronlu softmax aktivasyonlu bir çıkış katmanı içeren bir model oluşturun.
# Modeli categorical_crossentropy kayıp fonksiyonu ve Adam optimizasyon algoritması ile derleyin.
# Modeli 100 epoch boyunca, 8 verilik batch boyutu ile eğitin.
# Modelin eğitim ve test doğruluklarını hesaplayın ve yüzdelik olarak raporlayın.
# Not: Modelin özetini (summary()) ekleyerek yapısını inceleyin.
