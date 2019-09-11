from emnist import extract_training_samples, extract_test_samples
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os.path as path

if not path.isfile('Entrenado.model'):

	images_digits_training, labels_digits_training = extract_training_samples('digits')
	images_digits_test, labels_digits_test = extract_test_samples('digits')

	images_letters_training, labels_letters_training = extract_training_samples('letters')
	images_letters_test, labels_letters_test = extract_test_samples('letters')

	images_training = np.concatenate((images_digits_training, images_letters_training))
	images_test = np.concatenate((images_digits_test, images_letters_test))
	labels_training = np.concatenate((labels_digits_training, labels_letters_training))
	labels_test = np.concatenate((labels_digits_test, labels_letters_test))


	images_training = tf.keras.utils.normalize(images_training, axis=1)
	images_test = tf.keras.utils.normalize(images_test, axis=1)

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(62, activation=tf.nn.softmax))

	model.compile(optimizer='adam',
	             loss='sparse_categorical_crossentropy',
	             metrics=['accuracy'])

	model.fit(images_training, labels_training, epochs=3)
	
	model.save('Entrenado.model')

else:

	model = tf.keras.models.load_model('Entrenado.model')

	val_loss, val_acc = model.evaluate(images_test, labels_test)


	predictions = model.predict([images_test])

	print(np.argmax(predictions[1787]))



	plt.imshow(images_test[1787], cmap='binary')
	plt.show()



#plt.imshow(images_letters_test[152], cmap='binary')
#plt.show()
