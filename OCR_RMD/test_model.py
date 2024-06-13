'''
 * Copyright (c) 2014 Radhamadhab Dalai
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
'''

from tensorflow import keras
# from tensorflow.keras import layers

from tensorflow.python.keras import layers


# Define the RNN model
def build_ocr_rnn_model(input_shape, num_classes):
    model = keras.Sequential()

    # Add an RNN layer
    model.add(layers.SimpleRNN(units=128, input_shape=input_shape))

    # Add a fully connected layer
    model.add(layers.Dense(units=64, activation='relu'))

    # Add the output layer
    model.add(layers.Dense(units=num_classes, activation='softmax'))

    return model

# Define the input shape and number of classes for your OCR task

import tensorflow as tf
tf.config.run_functions_eagerly(True)

def train(train_images, train_labels, input_shape, num_classes , batch_size, epochs):
    model = build_ocr_rnn_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs)
    model.save('trained_ocr_model.h5')
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import SimpleRNN

def train2(input_data, train_labels, size, input_shape, num_classes, batch_size, epochs):
    # Create a Sequential model
    model = keras.Sequential()
    model.add(layers.SimpleRNN(units=size, input_shape=(size, 1)))
    model.add(layers.Dense(units=size, activation='softmax'))
    model.build(input_shape=(None, size, 1))
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(input_data, train_labels, batch_size=batch_size, epochs=epochs)
    # model.save('trained_ocr_model.h5')
    # Print the model summary
    model.summary()

    # Pass the reshaped input data to the model
    output = model.predict(input_data)
    print(output)

