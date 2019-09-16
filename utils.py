import os
import numpy as np
from imread import imread


def loadimgs(path, n = 0):

    X = []
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n

    for alphabet in os.listdir(path):

        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y, None]
        alphabet_path = os.path.join(path, alphabet)

        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images = []
            letter_path = os.path.join(alphabet_path, letter)

            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = imread(image_path)
                category_images.append(image)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            except ValueError as e:
                print(e)
                print("error - category_images: ", category_images)

            lang_dict[alphabet][1] = curr_y
            curr_y += 1
    y = np.vstack(y)
    X = np.stack(X)
    return X, y, lang_dict
