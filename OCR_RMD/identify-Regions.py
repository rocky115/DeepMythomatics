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
#
# # Load the image
image_path = '3.jpeg'

import cv2

# Load the image
# image_path = 'image.jpg'
image = cv2.imread(image_path)


# Define the number of lines
num_lines = 5

# Calculate the height of each line
image_height = image.shape[0]
line_height = image_height // num_lines

# Iterate through the lines and split the image
for i in range(num_lines):
    # Calculate the starting and ending y-coordinates for the line
    start_y = i * line_height
    end_y = (i + 1) * line_height

    # Extract the line region from the image
    line_image = image[start_y:end_y, :]

    # Display or process the line image
    cv2.imshow('Line Image', line_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
