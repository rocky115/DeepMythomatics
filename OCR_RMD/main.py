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
import cv2
import numpy as np
from model import model
from data_preprocessing import preprocess_image

class main():
    # Load and preprocess a sample image
    image_path = '3.png'
    image = cv2.imread(image_path)
    preprocessed_image = preprocess_image(image)
 # train images routinr
    import os
    import cv2
    import numpy as np

    # Path to the directory containing the training images
    train_images_dir = ''

    # List to store the training images and labels
    train_images = []
    train_labels = []

    # Iterate through the images in the directory
    for image_file in os.listdir(train_images_dir):
        if image_file.endswith('.png') or image_file.endswith('.jpg'):
            # Load the image
            image_path = os.path.join(train_images_dir, image_file)
            image = cv2.imread(image_path)

            # Preprocess the image if needed (resize, normalize, etc.)
            # ...

            # Extract the label from the image filename or any other method
            label = image_file.split('.')[0]

            # Append the preprocessed image and its label to the lists
            train_images.append(image)
            train_labels.append(label)

    # Convert the lists to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Print the shape of the train_images and train_labels arrays
    print('Train images shape:', train_images.shape)
    print('Train labels shape:', train_labels.shape)
