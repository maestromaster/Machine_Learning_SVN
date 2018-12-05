import imageio as imageio
import matplotlib.pyplot as plt
import itertools
from sklearn import datasets, svm, metrics
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy import misc
import cv2

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

digits = datasets.load_digits()
# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
print("n_samples", n_samples)
data = digits.images.reshape((n_samples, -1))
print("data[1] ", data[1])
print("digits.images[1] \n", digits.images[1])
exaples = np.arange(12).reshape(2, -1)
print("exaples ", exaples)

print("target[1] ", digits.target[1])
# Create a classifier: a support vector classifier

classifier = svm.SVC(gamma=0.001)
# We learn the digits on the first half of the digits

classifier.fit(data[:n_samples], digits.target[:n_samples])
# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]

predicted = classifier.predict(data[n_samples // 2:])

class_names = classifier.classes_

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))


cnf_matrix = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cnf_matrix)

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

# showing plot with confusion matrix
# np.set_printoptions(precision=2)
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')


# Predict one value
# img = Image.open("1.png")
# img.load()
# written_image_data = np.asarray(img, dtype="int8")
# print("written_image_data: ", written_image_data)

hand_digit = cv2.imread("6.png", cv2.IMREAD_GRAYSCALE)
hand_digit = misc.imresize(hand_digit, (8,8))
hand_digit = hand_digit.astype(digits.images.dtype)
hand_digit = misc.bytescale(hand_digit, high=16, low=0)
hand_digit = (16-hand_digit)

plt.figure()
plt.imshow(hand_digit, cmap=plt.cm.gray_r, interpolation='nearest')
print("hand_digit_data: ", hand_digit)

# digit_data = data[900].reshape(1,-1)
digit_data = hand_digit.reshape(1,-1)
digit_image = digit_data.reshape(8,-1)
predicted_item = classifier.predict(digit_data)
print("predicted_item: ", predicted_item)
print("digit_data: ", digit_data)

plt.figure()
plt.axis('off')
plt.imshow(digit_image, cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("Predicted %s: " % predicted_item)

plt.show()