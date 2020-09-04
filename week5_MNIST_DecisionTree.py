import struct
import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import itertools
from sklearn.tree import DecisionTreeClassifier


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


raw_train = read_idx(r"C:\Users\user\Desktop\DIP_Seminar\Machine Learning Seminar\week5\train-images-idx3-ubyte")

train_data = np.reshape(raw_train, (60000, 28 * 28))

train_label = read_idx(r"C:\Users\user\Desktop\DIP_Seminar\Machine Learning Seminar\week5\train-labels-idx1-ubyte")

# print(raw_train)
# print(raw_train.shape)
# print(train_label)
# print(train_label.shape)

raw_test = read_idx(r"C:\Users\user\Desktop\DIP_Seminar\Machine Learning Seminar\week5\t10k-images-idx3-ubyte")

test_data = np.reshape(raw_test, (10000, 28 * 28))

test_label = read_idx(r"C:\Users\user\Desktop\DIP_Seminar\Machine Learning Seminar\week5\t10k-labels-idx1-ubyte")

idx =  (train_label == 2) | (train_label == 3) | (train_label == 8)
X = train_data[idx]/255.0
Y = train_label[idx]
tree = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0).fit(X,Y)

idx =  (test_label == 2) | (test_label == 3) | (test_label == 8)
x_test = test_data[idx]/255.0
y_true = test_label[idx]
y_pred = tree.predict(x_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = metrics.confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, ["2", "3", "8"])

idx = np.where((y_pred == 2) & (y_true == 8))[0]
fig = plt.figure(figsize=(5, 30))

for i in range(len(idx)):
    ax = fig.add_subplot(len(idx), 1, i + 1)
    imgplot = ax.imshow(np.reshape(x_test[idx[i], :], (28, 28)), cmap=plt.cm.get_cmap("Greys"))
    imgplot.set_interpolation("nearest")
plt.show()
