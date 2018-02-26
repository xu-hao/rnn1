import keras
from rnn1_utils import MAXLEN, ctable, INVERT, chars, MODEL_PATH
from sys import argv
import numpy as np

model = keras.models.load_model(MODEL_PATH)

q = argv[1]
query = q + ' ' * (MAXLEN - len(q))

if INVERT:
    query = query[::-1]

x = np.zeros((1,MAXLEN, len(chars)), dtype=np.bool)
x[0] = ctable.encode(query, MAXLEN)
probas = model.predict(x)
guess = ctable.decode(probas[0], calc_argmax=True)
print(guess)

