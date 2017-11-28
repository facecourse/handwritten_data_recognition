import os
import cv2
import numpy as np
from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


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

directory_name='MNIST dataset'
image_container=[]
label_container=[]
for subfolder in os.listdir(directory_name):
    for images in os.listdir(directory_name+'/'+subfolder):
        image_loaded=cv2.imread(directory_name+'/'+subfolder+'/'+images,0)
        # cv2.imshow('dataset',image_loaded)
        # cv2.waitKey(10)
        image_feature=lbp_feature(image_loaded)
        image_container.append(image_feature)
        label_container.append(int(subfolder))



C_range = np.logspace(-2, 10, 2)
gamma_range = np.logspace(-9, 3, 2)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=10)
model = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
model.fit(image_container, label_container)
#
print("The best parameters are %s with a score of %0.2f"
      % (model.best_params_, model.best_score_))
model = LinearSVC(C=100.0, random_state=42)
model.fit(image_container, label_container)

# model= SVC(kernel='rbf',gamma=.1, C=10000)
# model.fit(image_container, label_container)

for images in os.listdir('testdata'):
    image_loaded=cv2.imread('testdata'+'/'+images,0)
    image_original = cv2.imread('testdata' + '/' + images, 1)
    image_feature=lbp_feature(image_loaded)
    prediction=model.predict(image_feature)[0]
    print"predicted image is of class: ",prediction
    cv2.putText(image_original, str(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 1)
    cv2.imshow('query image',image_original)
    cv2.waitKey(500000)



