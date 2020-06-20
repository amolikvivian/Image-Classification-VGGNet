#Initializing matplotlib backend to save plot
import matplotlib
matplotlib.use('Agg')

import os
import cv2
import pickle
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from imutils import paths
from networks.cnn import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

#Initializing Epochs and Batch Size
BS = 35
EPOCHS = 2

#Initializing Argument Parsing
ap = argparse.ArgumentParser()

ap.add_argument('-d', '--dataset', required = True,
			help = 'Path to input dataset of images')

ap.add_argument('-m', '--model', required = True, default = 'output/cnn.model', 
			help = 'Path to output trained model')

ap.add_argument('-l', '--label-bin', required = True, default = 'output/cnn_lbl.pickle', 
			help = 'Path to output label binarizer')

ap.add_argument('-p', '--plot', required = True, default = 'output/accuracy.png',
			help = 'Path to output accuracy/loss plot')

args = vars(ap.parse_args())

print("[Status] Loading images...")

data = []
labels = []

#Grabbing image path and shuffling order of images
imagePaths = sorted(list(paths.list_images(args['dataset'])))
random.seed(42)
random.shuffle(imagePaths)

#Looping over input images
for imagePath in imagePaths:

	#Reading and resizing images to 64x64 - required dimension for CNN
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))

	#Appending images to data list
	data.append(image)

	#Appending image labels to label list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

#Normalizing pixel density between [0, 1]
data = np.array(data, dtype = 'float') / 255.0
labels = np.array(labels)

#Randomly partitioning data into Train and Test (3:1)
(x_train, x_test, y_train, y_test) = train_test_split(data, labels,
				test_size = 0.25, random_state = 42)

#Converting labels as vectors - One Hot Encoding
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

#Image generator for data augmentation
aug = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.1,
	height_shift_range = 0.1, shear_range = 0.2, zoom_range = 0.2,
	horizontal_flip = True, fill_mode = 'nearest')

#Initializing VGGNet model
model = SmallVGGNet.build(width = 64, height = 64, depth = 3, classes = len(lb.classes_))

#Compiling Model
model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])

print("[Status] Training network...")
#Training the network
H = model.fit(x = aug.flow(x_train, y_train, batch_size = BS),
			validation_data = (x_test, y_test), steps_per_epoch = len(x_train) // BS,
			epochs = EPOCHS)

#Evaluating Network
print("[Status] Evaluating network...")
predictions = model.predict(x = x_test, batch_size = 32)
print(classification_report(y_test.argmax(axis = 1),
	predictions.argmax(axis = 1), target_names = lb.classes_))

#Plotting the training accuracy
N = np.arange(0, EPOCHS)
plt.style.use('ggplot')
plt.figure()
plt.plot(N, H.history['acc'], label = 'train_acc')
plt.plot(N, H.history['val_acc'], label = 'val_acc')
plt.title("Training Loss and Accuracy (net)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(args['plot'])

#Saving Model and Binarizer
print("[STATUS] Saving network and label binarizer...")
model.save(args['model'], save_format = 'h5')
f = open(args['label_bin'], 'wb')
f.write(pickle.dumps(lb))
f.close()