from PIL import Image
import multiprocessing as mp
import glob
import numpy as np
from setting import IMG_SIZE
import os

"""
This module loads the images in dogs_vs_cats dataset
After opening the images, it resize them to IMG_SIZE * IMG_SIZE
And normalize them and save them to numpy arrays cats.npy and dogs.npy
Note that these numpy arrays only contain images and not their labels

Assumptions:
    the data is in "train" directory in the current directory
"""
print("Importing images ...")
def import_image(path):
    """
    open an image given its path, resize it, normalize it (by dividing by 255),
    and convert it to a numpy array

    Args:
        path (string): the image full getPath

    Returns:
        The numpy array of the image where each value in the array is between 0-1
    """
    return np.array(Image.open(path).resize((IMG_SIZE,IMG_SIZE)), dtype = "float32")/255.


"""
the return type of glob.glob is a list of files
the reason to use
        os.path.join
 is to make the code more portable, otherwise on linux based system it is
    train/cat*.jpg
and on windows it is
    train\\cat*.jpg
  """

cats_to_import = glob.glob(os.path.join('..','train', 'cat*.jpg'))
dogs_to_import = glob.glob(os.path.join('..','train', 'dog*.jpg'))

"""
Creating pool of worker processes
 Later the map function, maps the function with one argument
 to an iterable which here is the cats (or dogs) image paths
 to each processor
 the result is a list of images that we convert it to a numpy array

"""
pool = mp.Pool()
cats = np.array(pool.map(import_image, cats_to_import))
print("cat array shape:", cats.shape)
np.save("cats.npy", cats)

dogs = np.array(pool.map(import_image, dogs_to_import))
print("dog array shape:", cats.shape)
np.save("dogs.npy", dogs)
