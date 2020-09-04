import struct
import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import itertools
import pywt
import cv2
from skimage.transform import resize

##
# import os
# import torch
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
#
# def listdir_fullpath(d):
#     return [os.path.join(d, f) for f in os.listdir(d)]
#
# class MNIST_manual(Dataset):
#     def __init__(self, image_dir):
#         self.image_dir = image_dir
#         self.folder_list = os.listdir(self.image_dir)
#
#         self.image_list = []
#         self.gt_list = []
#
#         for gt in self.folder_list:
#             locals()[str(gt)] = os.path.join(image_dir, str(gt))
#             self.image_list.extend(listdir_fullpath(locals()[str(gt)]))
#             self.gt_list.extend([gt] * len(os.listdir(locals()[str(gt)])))
#
#         self.preprocess = transforms.Compose([
#             transforms.ToTensor(),
#         ])
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def __getitem__(self, idx):
#         img = self.preprocess(Image.open(self.image_list[idx]))
#
#         return img, torch.tensor(int(self.gt_list[idx]))
# image_dir = "C:/Users/user/Desktop/mnist_png/mnist_png/testing"
#
# train_data = MNIST_manual(image_dir)
# train_loader = torch.utils.data.DataLoader(train_data,
#                                            batch_size=1,
#                                            shffle=False
#                                            )
#
# for i, (image, gt) in enumerate(train_loader):
#     print(image)
#     print(gt)
#     print(image.shape)
#     print(gt.shape)
##

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

raw_train = read_idx(r"C:\Users\user\Desktop\DIP_Seminar\Machine Learning Seminar\week5\train-images-idx3-ubyte")

train_data = np.reshape(raw_train, (60000, 28 *28))
# # cv2.imshow('d',train_data[0])
# # cv2.waitKey(0)
#
#
# for i in range(60000):
#      coeffs2 = pywt.dwt2(train_data[i],'bior1.3')
#      LL, (LH, HL, HH) = coeffs2
#      LL = np.array(LL).astype(np.uint8)
#      LL = LL[1:15, 1:15]
#      LH = np.array(LH).astype(np.uint8)
#      LH = LH[1:15, 1:15]
#
#      HL = np.array(HL).astype(np.uint8)
#      HL = HL[1:15, 1:15]
#      HH = np.array(HH).astype(np.uint8)
#      HH = HH[1:15, 1:15]
#      train_data[i] = np.vstack((np.hstack((LL,LH)),np.hstack((HL,HH))))
#      cv2.imshow('d', train_data[i])
#      cv2.waitKey(0)
#      if i % 1000 ==0:
#         print(i)
# train_data = np.reshape(train_data,(60000,28*28))
#
# train_data = np.reshape(raw_train,(60000,28*28))
train_label = read_idx(r"C:\Users\user\Desktop\DIP_Seminar\Machine Learning Seminar\week5\train-labels-idx1-ubyte")

# print(raw_train)
# print(raw_train.shape)
# print(train_label)
# print(train_label.shape)

raw_test = read_idx(r"C:\Users\user\Desktop\DIP_Seminar\Machine Learning Seminar\week5\t10k-images-idx3-ubyte")

test_data = np.reshape(raw_test, (10000, 28 *28))
#
# for i in range(10000):
#     coeffs2 = pywt.dwt2(test_data[i], 'bior1.3')
#     LL, (LH, HL, HH) = coeffs2
#     LL = np.array(LL).astype(np.uint8)
#     LL = LL[1:15, 1:15]
#     LH = np.array(LH).astype(np.uint8)
#     LH = LH[1:15, 1:15]
#     HL = np.array(HL).astype(np.uint8)
#     HL = HL[1:15, 1:15]
#     HH = np.array(HH).astype(np.uint8)
#     HH = HH[1:15, 1:15]
#     test_data[i] = np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))
#     if i % 1000 == 0:
#         print(i)
#test_data = np.reshape(test_data, (10000, 28 * 28))
#test_data = np.reshape(raw_test,(10000,28*28))

test_label = read_idx(r"C:\Users\user\Desktop\DIP_Seminar\Machine Learning Seminar\week5\t10k-labels-idx1-ubyte")

idx =  (train_label == 2) | (train_label == 3) | (train_label == 8)
X = train_data[idx]
Y = train_label[idx]
SVC = svm.SVC(kernel ='rfb',C=1,gamma='auto').fit(X,Y)

print('d')
idx =  (test_label == 2) | (test_label == 3) | (test_label == 8)
x_test = test_data[idx]
y_true = test_label[idx]
y_pred = SVC.predict(x_test)


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

idx = np.where((y_pred == 8) & (y_true == 3))[0]
fig = plt.figure(figsize=(5, 30))

for i in range(len(idx)):
    ax = fig.add_subplot(len(idx), 1, i + 1)
    imgplot = ax.imshow(np.reshape(x_test[idx[i], :], (28, 28)), cmap=plt.cm.get_cmap("Greys"))
    imgplot.set_interpolation("nearest")
plt.show()
