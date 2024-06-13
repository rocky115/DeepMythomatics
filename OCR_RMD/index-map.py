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
# Define the character index map
character_index_map = {}

# Add vowels
vowels = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ॠ', 'ऌ', 'ॡ', 'ए', 'ऐ', 'ओ', 'औ']
for i, vowel in enumerate(vowels):
    character_index_map[vowel] = i

# Add consonants
consonants = ['क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह']
for i, consonant in enumerate(consonants):
    character_index_map[consonant] = i + len(vowels)

# Add vowel signs and diacritics
vowel_signs = ['ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः', 'ँ', 'ऽ']
for i, vowel_sign in enumerate(vowel_signs):
    character_index_map[vowel_sign] = i + len(vowels) + len(consonants)

# Add numerals and digits
numerals = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९']
for i, numeral in enumerate(numerals):
    character_index_map[numeral] = i + len(vowels) + len(consonants) + len(vowel_signs)

# Print the character index map
for character, index in character_index_map.items():
    print(f"{character}: {index}")

# compound
character_index_map = {
    'श्री': 0,
    'क्षी': 1,
    'त्रि': 2,
    'ज्ञा': 3,
    # Add more compound characters as needed
}

## image resize
from PIL import Image

image = Image.open('3.jpeg')
print(f"Original size : {image.size}")

sunset_resized = image.resize((64, 64))
sunset_resized.save('3.jpeg')