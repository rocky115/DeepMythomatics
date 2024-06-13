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

# Example list of training labels in Sanskrit
train_labels = ["श्रीमद्भगवद्गीता", "महाभारत", "संस्कृत"]

# Create a mapping from characters to numerical indices
characters = sorted(set(''.join(train_labels)))
char_to_index = {char: index for index, char in enumerate(characters)}

# Convert the labels to numerical indices using the mapping
train_labels_encoded = []
for label in train_labels:
    encoded_label = [char_to_index[char] for char in label]
    train_labels_encoded.append(encoded_label)

# Convert the encoded labels to numpy array
train_labels_encoded = np.array(train_labels_encoded)

# Print the encoded labels
print('Encoded Labels:', train_labels_encoded)
