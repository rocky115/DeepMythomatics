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

# Define the input shape and number of classes for your Sanskrit OCR task
input_shape = (image_height, image_width, num_channels)
num_classes = 10  # Example: for OCR of 10 Sanskrit characters

# Build the RNN OCR model
model = build_ocr_rnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 10

model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs)

# Evaluate the model
val_loss, val_acc = model.evaluate(val_images, val_labels)
print('Validation Loss:', val_loss)
print('Validation Accuracy:', val_acc)

# Save the trained model
model.save('trained_ocr_model.h5')
# eitaa dekho