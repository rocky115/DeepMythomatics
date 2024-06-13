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

import tensorflow as tf
# from tensorflow.keras import l
from keras import layers
# Define the RNN model architecture
def model(image_height, image_width):

        model = tf.keras.Sequential([
            layers.SimpleRNN(units=128, input_shape=(image_height, image_width), activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        # Train the model
        model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

        # Evaluate the model
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)




from tensorflow import keras
from tensorflow.keras import layers

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
input_shape = (image_height, image_width, num_channels)
num_classes = 26  # Example: for OCR of English alphabet

# Build the RNN OCR model
model = build_ocr_rnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
