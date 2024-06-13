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

import numpy as np
from PIL import Image

# Example list of image file paths for training
train_image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']

# Define the image dimensions
image_height = 32
image_width = 32
num_channels = 3

# Load and preprocess the images
train_images = []
for image_path in train_image_paths:
    image = Image.open(image_path)
    image = image.resize((image_width, image_height))
    image = np.array(image)
    train_images.append(image)

# Convert the train_images list to a numpy array
train_images = np.array(train_images)

# Normalize the pixel values to the range of 0 to 1
train_images = train_images.astype('float32') / 255.0

# Print the shape of the train_images array
print('Shape of train_images:', train_images.shape)
