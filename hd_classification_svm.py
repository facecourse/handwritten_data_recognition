import os
import cv2
import numpy as np
from skimage import feature
from sklearn.svm import LinearSVC

def lbp_feature(image):
    lbp = feature.local_binary_pattern(image, 24,
                                       8, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, 8 + 3),
                             range=(0, 8 + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 0.001)

    return hist

directory_name='Handwritten_data'
image_container=[]
label_container=[]
for subfolder in os.listdir(directory_name):
    if subfolder=='0':
        for images in os.listdir(directory_name+'/'+subfolder):
            image_loaded=cv2.imread(directory_name+'/'+subfolder+'/'+images,0)
            image_feature=lbp_feature(image_loaded)
            image_container.append(image_feature)
            label_container.append(0)
    if subfolder == '1':
        for images in os.listdir(directory_name + '/' + subfolder):
            image_loaded = cv2.imread(directory_name + '/' + subfolder + '/' + images, 0)
            image_feature=lbp_feature(image_loaded)
            image_container.append(image_feature)
            label_container.append(1)



model = LinearSVC(C=100.0, random_state=42)
model.fit(image_container, label_container)

for images in os.listdir('testdata'):
    image_loaded=cv2.imread('testdata'+'/'+images,0)
    image_original = cv2.imread('testdata' + '/' + images, 1)
    image_feature=lbp_feature(image_loaded)
    prediction=model.predict(image_feature)[0]
    print"predicted image is of class: ",prediction
    cv2.putText(image_original, str(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 1)
    cv2.imshow('query image',image_original)
    cv2.waitKey(5000)



