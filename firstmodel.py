from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

hist = model.fit(X_train, y_train, batch_size=200, epochs=30,
          validation_split=0.2, verbose=2, shuffle=True)

model.save('firstmodel.h5')

score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

print('The final accuracy of model %.4f%%' % accuracy)
