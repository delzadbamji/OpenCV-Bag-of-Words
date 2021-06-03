import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def showHist(orb,TrainImPath):
    for i in [1,24,35,55]:
        im = cv2.imread(TrainImPath[i])
        kp = orb.detect(im, None)
        kp, des = orb.compute(im, kp)
        print("please wait while we compute a sample keypoint description...")
        print("The program will proceed after the image tab is closed.")
        img = draw_keypoints(im, kp,i)

def dataset_creation(directoryString, TestorTrain):
    imPath = []
    imClass = []

    for training_name in TestorTrain:
        directory = os.path.join(directoryString, training_name)
        Cpath = file_list_func(directory)
        imPath += Cpath
    # print(len(TestorTrain))

    lenImpath = len(imPath)
    lenTestorTrain = len(TestorTrain)
    for i in range(0, lenTestorTrain):
        imClass += [i] * (lenImpath // lenTestorTrain)

    D = []

    for i in range(lenImpath):
        D.append((imPath[i], imClass[i]))

    return D, imPath, imClass


def file_list_func(path):
    return (os.path.join(path, f) for f in os.listdir(path))


def draw_keypoints(vis, keypoints,i, color = (0, 255, 255)):
    for kp in keypoints:
            x, y = kp.pt
            plt.imshow(cv2.circle(vis, (int(x), int(y)), 2, color))
    plt.title('keypoints detected')
    plt.show()
    plt.hist(vis.ravel(),256,[0,256])
            # show the plotting graph of an image
    # plt.plot(histr)
    plt.title('histogram')
    plt.show()


Training = os.listdir('train')
Testing = os.listdir('test')


train, TrainImPath, TrainImClass = dataset_creation("train",Training)
test, TestImPath, TestImClass = dataset_creation("test",Testing)

image_paths, y_train = zip(*train)
image_paths_test, y_test = zip(*test)

des_list = []

orb = cv2.ORB_create()
# print(TrainImPath)

showHist(orb,TrainImPath)
print("please wait while we compute the classes...")

for image_pat in TrainImPath:
    im = cv2.imread(image_pat)
    kp = orb.detect(im, None)
    keypoints, descriptor = orb.compute(im, kp)
    des_list.append((image_pat, descriptor))

descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# print(descriptors.shape)

descriptors_float = descriptors.astype(float)
k = 200
voc, variance = kmeans(descriptors_float, k, 1)

im_features = np.zeros((len(TrainImPath), k), "float32")
for i in range(len(TrainImPath)):
    words, distance=vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

stdslr = StandardScaler().fit(im_features)
im_features = stdslr.transform(im_features)

clf = LinearSVC(max_iter=80000)
clf.fit(im_features, np.array(y_train))


des_list_test = []

for image_pat in image_paths_test:
    image = cv2.imread(image_pat)
    kp = orb.detect(image, None)
    keypoints_test, descriptor_test = orb.compute(image, kp)
    des_list_test.append((image_pat, descriptor_test))


test_features=np.zeros((len(image_paths_test), k), "float32")
for i in range(len(image_paths_test)):
    words,distance = vq(des_list_test[i][1], voc)
    for w in words:
        test_features[i][w] += 1

# print(test_features)
test_features=stdslr.transform(test_features)

true_classes = []

for i in y_test:
    if i == 0:
        true_classes.append("accordian")
    elif i == 1:
        true_classes.append("dollar bill")
    elif i == 2:
        true_classes.append("motorbike")
    elif i == 3:
        true_classes.append("Soccer ball")

predict_classes = []
for i in clf.predict(test_features):
    if i == 0:
        predict_classes.append("accordian")
    elif i == 1:
        predict_classes.append("dollar bill")
    elif i == 2:
        predict_classes.append("motorbike")
    elif i == 3:
        predict_classes.append("Soccer ball")

print("true dataset: ", true_classes)
print("predicted dataset: ", predict_classes)

clf.predict(test_features)

accuracy = accuracy_score(true_classes, predict_classes)
print("model accuracy is: ", accuracy)
