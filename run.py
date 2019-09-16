import batcher
import itertools
from keras.optimizers import Adam
import numpy as np
import pickle
import re
import siamese_net
import sys
import time
import utils


args = list(itertools.filterfalse(lambda x: re.match('^--', x), sys.argv))

if re.match('load', args[1]):

    train_folder = "./images_background"
    test_data = "./images_evaluation"

    X, y, c = utils.loadimgs(train_folder)

    X.dump("./tensors/X.np.dat")
    y.dump("./tensors/y.np.dat")
    with open("./tensors/c.dat", 'wb') as pickle_file:
        pickle.dump(c, pickle_file)

if re.match('learn', args[1]):

    X = np.load("./tensors/X.np.dat")
    y = np.load("./tensors/y.np.dat")
    with open("./tensors/c.dat", "rb") as pickle_file:
        c = pickle.load(pickle_file)

    model = siamese_net.get_model((105, 105, 1))
    optimizer = Adam(lr = 0.00006)
    model.compile(loss="binary_crossentropy", optimizer = optimizer)

    print("Starting training...")

    t_start = time.time()
    for i in range(1, 2000 + 1):
        (inputs, targets) = batcher.get_batch(32, X, y, c)
        loss = model.train_on_batch(inputs, targets)

        if i % 200 == 0:

            print("\n ------------- \n")
            print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
            print("Train Loss: {0}".format(loss))

            model.save_weights(os.path.join(mode_path, "weights.{}.h5".format(i)))
