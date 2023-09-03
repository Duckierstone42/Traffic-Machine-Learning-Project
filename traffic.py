import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from sklearn.model_selection import train_test_split
tf.config.set_soft_device_placement(True)
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43  #Change this based on the data set used
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    """
    images=[] #List of images
    labels=[] #List of types for each image
    files = listdir_nohidden(data_dir) #First error--Something wrong with the function
    for category_folder in files: #Need to add another for loop here, need to go one thing in
        #Add the index of the file, which also happens to be the category
        
        examples = listdir_nohidden(os.path.join(data_dir,category_folder))
        for example in examples:

            labels.append(int(category_folder))

            image = cv2.imread(os.path.join(data_dir,category_folder,example),1) #Error: This is not finding a file in this location
            
            #Giving me this file path: gtsrb-small\0 Which appears to not work
            
            image = cv2.resize(image,(IMG_WIDTH,IMG_HEIGHT)) 
            images.append(image)
    return (images, labels)




def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential()
    #Add convulational and pooling layers
    model.add(Conv2D(30, (3,3), activation="relu",input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(40, (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(200,activation="relu"))
    model.add(Dropout(.2))
    #Final layer
    model.add(Dense(NUM_CATEGORIES,activation="sigmoid"))

    model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=["accuracy"])
    return model
if __name__ == "__main__":
    main()
#Code now runs, but accuracy appears to be severly limited when using gtsrb. Not increasing in acuracy.