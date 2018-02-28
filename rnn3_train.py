'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
from datetime import datetime
from time import time

from keras.callbacks import LambdaCallback, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
import io
import re
import rnn3_utils as utils

path = sys.argv[1]
start_seq = sys.argv[2]

with io.open(path, encoding='utf-8') as f:
    text = f.read()

files = [m.start() for m in re.finditer(re.escape(start_seq), text)]
print('corpus length:', len(text))
print("number of files: ", len(files))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

batch_size = 128
seq_len = 64
step_size = utils.step
# cut the text in semi-redundant sequences of maxlen characters
seqs = []
next_chars = []
for i in range(0, len(text) - seq_len, step_size):
    seqs.append(text[i:i + seq_len])
    next_chars.append(text[i + seq_len])

# truncate number of samples to multiples of batch_size
# trunc = len(seqs) // batch_size * batch_size
# seqs = seqs[:trunc]
# next_chars = next_chars[:trunc]

print('nb sequences:', len(seqs))

print('Vectorization...')
x = np.zeros((len(seqs), seq_len, len(chars)), dtype=np.bool)
y = np.zeros((len(next_chars), len(chars)), dtype=np.bool)
for i, seq in enumerate(seqs):
    for t, char in enumerate(seq):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, input_shape=(seq_len, len(chars)), return_sequences=False, stateful=False))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def y_to_char(diversity, preds):
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]
    return next_char


def seq_to_X(seq):
    X = np.zeros((1, seq_len, len(chars)))
    for t, char in enumerate(seq):
        X[0, t, char_indices[char]] = 1.
    return X



def on_epoch_end(epoch, logs):
    if (epoch + 1) % 5 != 0:
        return
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        index = np.random.choice(files)
        seq = text[index:index + seq_len]

        generated = seq
        print('----- Generating with seed: "' + seq + '"')
        sys.stdout.write(generated)

        for i in range(400):
            X = seq_to_X(seq)
            y = model.predict(X, verbose=0)[0]
            next_char = y_to_char(diversity, y)

            generated += next_char
            seq = seq[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
tensorboard = TensorBoard(log_dir="logs3/{}".format(datetime.now()))

model.fit(x, y,
          batch_size=batch_size,
          epochs=60,
          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end),tensorboard])
