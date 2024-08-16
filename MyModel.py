import keras
from keras.api import layers
import numpy as np


class NNetwork:
    def __init__(self, inputs: int,
                 *my_layers: tuple):
        self.my_layers = my_layers
        self.inputs = inputs

        inp = layers.Input(shape=( 1, inputs))
        x = layers.Dense(my_layers[0], activation='relu', dtype='float32')(inp)
        for i in range(1, len(my_layers)-1):
            x = layers.Dense(my_layers[i], activation='relu', dtype='float32')(x)
        out1 = layers.Dense(my_layers[-1], activation='softmax', dtype='float32')(x)
        out2 = layers.Dense(my_layers[-1], activation='softmax', dtype='float32')(x)
        self.model = keras.api.Model(inp, [out1, out2])

    def __call__(self, inputs):
        return self.model(inputs)

    def get_total_weight(self):
        return getLength(self.model.get_weights())

    def set_weight(self, weights):
        b = []
        off = 0
        for i in range(len(self.my_layers)):
            if i == 0:
                b.append(np.ndarray(shape=(self.inputs, self.my_layers[0],), buffer=np.array(weights[:self.my_layers[0] * self.inputs]), dtype='float32'))
                off += self.my_layers[0] * self.inputs
                b.append(np.ndarray(shape=(self.my_layers[0], ), buffer=np.array(weights[off:off+self.my_layers[0]]), dtype='float32'))
                off += self.my_layers[0]
            else:
                b.append(np.ndarray(shape=(self.my_layers[i - 1], self.my_layers[i],),
                                    buffer=np.array(weights[off: off + self.my_layers[i] * self.my_layers[i - 1]]),
                                    dtype='float32'))
                off += self.my_layers[i] * self.my_layers[i - 1]
                b.append(np.ndarray(shape=(self.my_layers[i],),
                                    buffer=np.array(weights[off:off + self.my_layers[i]]),
                                    dtype='float32'))
                off += self.my_layers[i]

        b.append(np.ndarray(shape=(self.my_layers[i - 1], self.my_layers[i],),
                            buffer=np.array(weights[off: off + self.my_layers[i] * self.my_layers[i - 1]]),
                            dtype='float32'))
        off += self.my_layers[i] * self.my_layers[i - 1]
        b.append(np.ndarray(shape=(self.my_layers[i],),
                            buffer=np.array(weights[off:off + self.my_layers[i]]),
                            dtype='float32'))
        self.model.set_weights(b)




def getLength(element):
    if isinstance(element, list) or isinstance(element, np.ndarray):
        return sum([getLength(i) for i in element])
    return 1
