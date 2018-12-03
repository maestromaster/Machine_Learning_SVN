import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

print(digits.data.shape)

plt.gray()
plt.matshow(digits.images[500])
plt.show()