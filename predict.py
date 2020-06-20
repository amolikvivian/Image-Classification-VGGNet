import cv2
import pickle
import argparse
from keras.models import load_model

#Initializing Argument Parsing
ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required = True,
	help = 'Path to input image we are going to classify')

ap.add_argument('-m', '--model', required = True,
	help = 'Path to trained Keras model')

ap.add_argument('-l', '--label-bin', required = True,
	help = 'Path to label binarizer')

args = vars(ap.parse_args())

#Loading image and resizing
image = cv2.imread(args['image'])
img = cv2.imread(args['image'])
image = cv2.resize(image, (64, 64))

#Normalizing pixel densities between [0, 1]
image = image.astype('float') / 255.0

#Reshaping image to batch dimension
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

#Loading Model and Label Binarizer
print('[STATUS] Loading network and label binarizer...')
model = load_model(args['model'])
lb = pickle.loads(open(args['label_bin'], 'rb').read())

#Making prediction on input image
pred = model.predict(image)

#Finding class with maximum probabilty
id = pred.argmax(axis = 1)[0]
label = lb.classes_[id]

#Drawing class name and probablity percentage on original image
text = "{}: {:.2f}%".format(label, pred[0][id] * 100)
cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(0, 255, 100), 2)

#Displaying image with prediction
cv2.imshow("Image", img)
cv2.waitKey(0)