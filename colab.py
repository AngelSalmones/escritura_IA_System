# Imports
from emnist import extract_training_samples, extract_test_samples
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras
import os.path as path

# Variables
batch_size = 128
num_classes = 62
epochs = 5

# Input image dimensions
img_rows, img_cols = 28, 28

# The data, split between train and test sets
images_digits_training, labels_digits_training = extract_training_samples('digits')
images_digits_test, labels_digits_test = extract_test_samples('digits')

images_letters_training, labels_letters_training = extract_training_samples('letters')
images_letters_test, labels_letters_test = extract_test_samples('letters')

images_training = np.concatenate((images_digits_training, images_letters_training))
images_test = np.concatenate((images_digits_test, images_letters_test))
labels_training = np.concatenate((labels_digits_training, labels_letters_training))
labels_test = np.concatenate((labels_digits_test, labels_letters_test))



# Reshaping images acording to the format in keras
if K.image_data_format() == 'channels_first':
    images_training = images_training.reshape(images_training.shape[0], 1, img_rows, img_cols)
    images_test = images_test.reshape(images_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    images_training = images_training.reshape(images_training.shape[0], img_rows, img_cols, 1)
    images_test = images_test.reshape(images_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)



images_training = images_training.astype('float32')
images_test = images_test.astype('float32')



images_training /= 255
images_test /= 255



print('images_training shape:', images_training.shape)
print(images_training.shape[0], 'train samples')
print(images_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
labels_training = keras.utils.to_categorical(labels_training, num_classes)
labels_test = keras.utils.to_categorical(labels_test, num_classes)

if not path.isfile('siquiero.model'):


    # Creating the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    #Training the model
    model.fit(images_training, labels_training,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(images_test, labels_test))

    model.save('siquiero.model')


else:

    model = tf.keras.models.load_model('siquiero.model')

# Model Score
# val_loss, val_acc = model.evaluate(images_test, labels_test)
# predictions = model.predict([images_test])
# print(np.argmax(predictions[1787]))
# #plt.imshow(images_test[1787], cmap='binary')
# #plt.show()


#val_loss, val_acc = model.evaluate(images_test, labels_test)

my_image = plt.imread('A2.jpeg')



from skimage.transform import resize
from skimage import color, io
#my_image = io.imread('A.jpeg')
my_image_resized = resize(my_image, (28,28,1))


#print(my_image_resized)

unos = np.ones((28, 28, 1), dtype='float32')

prueba = np.subtract(unos, my_image_resized)



#plt.imshow(np.squeeze(my_image_resized), cmap='binary')
#plt.show()

#print(prueba)






predictions = model.predict(np.array( [prueba,] ))
print(np.argmax(predictions[0]))
#plt.imshow(color.rgb2gray(images_test[0]), cmap='binary')
#plt.imshow(np.squeeze(img), cmap='binary')
#plt.imshow(np.squeeze(my_image_resized), cmap='binary')
#plt.show()


plt.imshow(np.squeeze(prueba), cmap='binary')
plt.show()

#print('Test loss:', val_loss)
#print('Test accuracy:', val_acc)

