import json
import math

from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

modelname = "./doge_trainer_"+str(datetime.now())

pi_data = []
with open('pressure_index_train_data.txt','r') as pi:
    pi_data = json.load(pi)

price_data = []
with open('prices_test_data.txt','r') as pr:
    price_data = json.load(pr)

pi_frames = []
outcomes = []
buys = 0
sells = 0
for i in range(45,len(pi_data)-80):
    pi_frames.append(pi_data[i-45:i])
    outcomes.append([0,0,1])
    current_price = price_data[i]
    for j in range(1,80):
        next_price = price_data[i+j]
        change = (next_price-current_price)/current_price
        if change >= 0.02:
            buys+=1
            outcomes[i-45] = [1,0,0]
            break
        elif change <= -0.02:
            sells+=1
            outcomes[i-45] = [0,1,0]
            break

print(buys)
print(sells)

pi_frames = np.array(pi_frames)
pi_frames = pi_frames.reshape(len(pi_frames),45)
outcomes = np.array(outcomes)
outcomes = outcomes.reshape(len(outcomes),3)

print(pi_frames.shape)
print(outcomes.shape)

model = keras.Sequential()
model.add(layers.Dense(90,batch_input_shape=(1,45,),activation="tanh"))
model.add(layers.Dropout(0.2))
model.add(layers.Reshape(target_shape=(1,90)))
model.add(layers.GRU(90,input_shape=(1,90),stateful=True,return_sequences=True,activation="tanh"))
model.add(layers.Reshape(target_shape=(90,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(60,activation="tanh"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(3,activation="softmax"))
# model.add(layers.Reshape(target_shape=(3,)))

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=["accuracy"],
)

history = model.fit(pi_frames, outcomes, batch_size=1, epochs=256, validation_split=0)

# test_scores = model.evaluate(pi_frames, outcomes, verbose=2)
# print("Test loss:", test_scores[0])
# print("Test accuracy:", test_scores[1])




model.save(modelname,save_format='tf')
