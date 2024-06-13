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

validation_text_file = 'slice-1.txt'




def prepare_labels(validation_text_file):
    val_labels = []

    with open(validation_text_file, 'r') as file:
        for line in file:
            print(line)
            # line = line.strip()

            label = line.split(" ")
            print(label)
            val_labels.append(label)

# val_labels = np.array(val_labels)

# Print the number of validation labels
#     print('labels:', val_labels)


#
# # Example list of training labels in Sanskrit
# train_labels = ["श्रीमद्भगवद्गीता", "महाभारत", "संस्कृत"]
#
# # Create a mapping from characters to numerical indices
# val_labels = ["Hello", "Ice", "table"]
# val_labels = [['धन्यवादान्ददानः', 'महाभारत', 'संस्कृत']]
    val_labels = val_labels[0]
    characters = sorted(set(''.join(val_labels)))
    print(characters)

    char_to_index = {char: index for index, char in enumerate(characters)}

    # Convert the labels to numerical indices using the mapping
    train_labels_encoded = []
    for label in val_labels:
        encoded_label = [char_to_index[char] for char in label]
        # encoded_label = [char_to_index[char] for char in label]
        train_labels_encoded.append(encoded_label)
        # train_labels_encoded.append(encoded_label)

    # Convert the encoded labels to numpy array
    arr = []
    for v in train_labels_encoded:
        for val in v:
            arr.append(val)


    print(arr)
    # train_labels_encoded = np.array(train_labels_encoded)
    train_labels_encoded = np.array(arr)
# Print the encoded labels
#     print('Encoded Labels:', train_labels_encoded)
#
#
#
    return train_labels_encoded


character_index_map = {
    'अ': 0,
    'आ': 1,
    'इ': 2,
    'ई': 3,
    'उ': 4,
    'ऊ': 5,
    'ऋ': 6,
    'ए': 7,
    'ऐ': 8,
    'ओ': 9,
    'औ': 10,
    'क': 11,
    'ख': 12,
    'ग': 13,
    # Add more characters as needed
}
# char = 'अ'
# char_index = character_index_map.get(char)
# print(char_index)  # Output: 0