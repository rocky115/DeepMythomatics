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
from data_preparation  import data_process

from test_model import build_ocr_rnn_model
from test_model import train
from test_model import train2
from prepare_labels import prepare_labels
import cv2
import numpy as np

dataset_dir = 'slice33'

class test():

    list =  data_process(dataset_dir)

    validation_text_file = 'slice33/slice-33.txt'

    train_imgs = list[0]
    val_imgs = list[1]
    test_imgs = list [2]

    train_labels = list[3]
    val_labels = list[4]
    test_labels = list[5]

    train_labels = prepare_labels(validation_text_file)

    # list = [1 , 2, 3, 4]
    # train_labels = np.array(list)

    print(len(train_labels))

    # Train the model
    batch_size =1
    epochs = 10
    my_array = np.array([[1.1, 2.2, 3.3]], dtype=np.float)

    # Prepare your training data (images and labels)
    # train_images = ""
    # train_labels = ""

    # Example input data
    # input_data = np.array([[1, 2, 3]])  # Shape: (1, 3)
    #
    # # Reshape the input data to match the expected shape
    # input_data = np.reshape(input_data, (1, 3, 1))

   #input image
    #
    # image = cv2.imread('/home/radha/Works/OCR/Sanskrit_OCR/impls/Sans-Sansan/slice33/slice-33.jpg')
    # # image = image / 255.0
    # height, width, _ = image.shape
    # # Convert the image to a NumPy array
    # image_array = np.array(image, dtype=np.float)
    #
    # input_data = np.reshape(image_array, (1, height*width, 1))

    # input_shape = (1, 30)

    size = 27000

    # Create a NumPy array of random float values between 0.1 and 1
    array = np.random.uniform(0.1, 1, size)

    array2 = np.random.uniform(1, 15, size)
    my_array = np.array([array2], dtype=np.float)
    input_data = np.array([[array]])
    input_data = np.reshape(input_data, (1, size, 1))
    input_shape = (1, size)
    # input_shape = (1, height*width)
    num_classes = 3  # Example: for OCR of Sanskrit alphabet

    # model = build_ocr_rnn_model(input_shape, num_classes)
    # train(train_imgs, train_labels, input_shape, num_classes, batch_size, epochs)
    # train(my_array, my_array, input_shape, num_classes, batch_size, epochs)
    # train2(input_data, my_array, input_shape, num_classes, batch_size, epochs)

    train2(input_data, my_array, size, input_shape, num_classes, batch_size, epochs)


test()

#
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import SimpleRNN
#
# # Example input data
# input_data = np.array([[1, 2, 3]])  # Shape: (1, 3)
#
# # Reshape the input data to match the expected shape
# input_data = np.reshape(input_data, (1, 3, 1))  # Shape: (1, 3, 1)
#
# # Create a Sequential model
# model = Sequential()
# model.add(SimpleRNN(units=32, input_shape=(3, 1)))
#
# # Print the model summary
# model.summary()
#
# # Pass the reshaped input data to the model
# output = model.predict(input_data)
# print(output)
#
#
#
