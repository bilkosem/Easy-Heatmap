import cv2
import numpy as np

# Kmeans color segmentation
def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

# Load image and perform kmeans
#image = cv2.imread('1.jpg')
image = cv2.imread(r'D:\borders.jpg')
#kmeans = kmeans_color_quantization(image, clusters=2)

h, w = image.shape[:2]
samples = np.zeros([h*w,3], dtype=np.float32)
count = 0

for x in range(h):
    for y in range(w):
        samples[count] = image[x][y]
        count += 1

compactness, labels, centers = cv2.kmeans(samples,
        2, 
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
        1, 
        cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
res = centers[labels.flatten()]
result = res.reshape((image.shape))

# Floodfill
seed_point = (100, 115)
cv2.floodFill(result, None, seedPoint=seed_point, newVal=(36, 255, 12), loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))
result.get
cv2.imshow('image', image)
cv2.imshow('result', result)
cv2.waitKey()     

import matplotlib.pyplot as plt
plt.imshow(result)
plt.xticks([]), plt.yticks([])
plt.show()