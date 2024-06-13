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

import os
import cv2
import numpy as np
from data_preprocessing import preprocess_image

def data_process(dataset_dir):


    list = []

# Set the path to your dataset directory
#         dataset_dir = 'path_to_dataset_directory'

        # Initialize lists for storing images and corresponding labels
    images = []
    labels = []

        # Iterate through the dataset directory
    for root, dirs, files in os.walk(dataset_dir):
    # for files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):

                    # Read the image
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = preprocess_image(image)

                    # Preprocess the image (resize, normalize, etc.) as needed
                    # image = preprocess_image(image)
                label = file.split('.')[0]

                    # Append the preprocessed image and its label to the lists
                    # val_images.append(image)
                    # val_labels.append(label)

                    # Extract the label from the image file name or directory structure
                    # label = extract_label(file)

                    # Append the image and label to the lists

                images.append(image)
                labels.append(label)

        # Convert the image and label lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

        # Split the dataset into training, validation, and testing sets
    train_images, val_images, test_images = np.split(images, [int(0.7 * len(images)), int(0.85 * len(images))])
    train_labels, val_labels, test_labels = np.split(labels, [int(0.7 * len(labels)), int(0.85 * len(labels))])

    list.insert(0,train_images)
    list.insert(1,val_images)
    list.insert(2,test_images)

    list.insert(3, train_labels)
    list.insert(4, val_labels)
    list.insert(5, test_labels)


    return list
