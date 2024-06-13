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
from PIL import Image

def slice_image(image, slice_width, slice_height):
    image_width, image_height = image.size
    slices = []

    for y in range(0, image_height, slice_height):
        for x in range(0, image_width, slice_width):
            slice_box = (x, y, x + slice_width, y + slice_height)
            slice_image = image.crop(slice_box)
            slices.append(slice_image)

    return slices

# Example usage
image_path = 'data_G_test/page-006.jpeg'  # Path to your image
slice_width = 30  # Width of each slice
slice_height = 30  # Height of each slice

# Load the image
image = Image.open(image_path)

# Slice the image into smaller regions
sliced_images = slice_image(image, slice_width, slice_height)

# Process the sliced images (e.g., apply OCR on each slice)

# Example: Save the sliced images
for i, sliced_image in enumerate(sliced_images):
    sliced_image.save(f'slice_{i}.jpg')
