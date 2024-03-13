
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re


window_size = 5
batch_size = 1

with open('trainingdata_min.txt', 'r', encoding='utf8') as f:
    # data = set(f.read().split())
    data = re.findall(r"[\w']+|[.,!?;]", f.read().lower())
    uniqueData = list(set(data))
    dataDict = {word: index for index, word in enumerate(uniqueData)}

# data = ["Hei", "jeg", "heter", "nico", "og", "jeg", "er", "kul"]
vocab_size = len(uniqueData)
dataSize = vocab_size - window_size
X_train = []
Y_train = []
for i in range(dataSize):
    X = tuple(tf.one_hot(dataDict[data[i+j]], vocab_size) for j in range(window_size))
    Y = tf.one_hot(dataDict[data[i+window_size]], vocab_size)

    X_train.append(X)
    Y_train.append(Y)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Shuffle the training data
idx = np.random.permutation(dataSize)
X_train, Y_train = X_train[idx], Y_train[idx]


# X_train = X_train.reshape((dataSize, window_size, 1))
# Y_train = Y_train.reshape((dataSize, vocab_size))
# Y_train = Y_train.reshape((dataSize,))

print("datasize: ", dataSize)

# Sneak peek at data:
# for w in X_train[0]:
#     print(w[0])
# print(Y_train[0])

model = tf.keras.Sequential([
    # tf.keras.layers.Embedding(vocab_size, 64, input_length=window_size),
    # tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.LSTM(units=32, input_shape=(window_size, vocab_size)),
    tf.keras.layers.Dense(units=vocab_size),
    tf.keras.layers.Softmax()
])

# loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
# model.compile(loss=loss, optimizer='adam')
model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(), metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.RMSprop(learning_rate=0.02), metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=8, batch_size=batch_size, verbose=True, validation_split=0.05)
model.save("shittygpt.model")

# print(history)
# print(history.history)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

