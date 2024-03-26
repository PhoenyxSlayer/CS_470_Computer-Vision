import numpy as np
import cv2
from skimage.segmentation import slic, mark_boundaries

def calc_dist(data, centroid):
    centroid = np.array(centroid, dtype="float32")
    centroid = np.reshape(centroid, (1,3))
    data = data.astype("float32")
    data = data - centroid
    data = np.square(data)
    data = np.sum(data, axis=1)
    data = np.sqrt(data)
    return data

def pick_within_thresh(data, centroid, dist):
    data = calc_dist(data, centroid)
    data = np.where(data < dist, np.ones(data.shape), np.zeros(data.shape))
    data = np.uint8(data)
    return data

def pick_closest(data, centroid):
    all_dist = calc_dist(data, centroid)
    chosen_index = np.argmin(all_dist)
    return chosen_index

def find_WBC(image):
    segments = slic(image, n_segments=50, sigma=5, start_label=0, compactness=10)
    cnt = len(np.unique(segments))
    group_means = np.zeros((cnt, 3), dtype="float32")
    for specific_group in range(cnt):
        mask_image = np.where(segments == specific_group, 255, 0).astype("uint8")
        mask_image = np.expand_dims(mask_image, axis=-1)
        group_means[specific_group] = cv2.mean(image, mask=mask_image)[0:3]

    k = 3
    _, bestLabels, centers = cv2.kmeans(group_means.astype("float32"),  k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    chosen_index = pick_closest(centers, (84, 44, 254))
    new_centers = np.zeros_like(centers)
    new_centers[chosen_index] = (255, 255, 255)
    new_centers = np.uint8(new_centers)
    new_colors = new_centers[bestLabels.flatten()]
    cell_mask = new_colors[segments]

    cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
    
    cells = []

    retval, labels = cv2.connectedComponents(cell_mask, None, connectivity=8, ltype=cv2.CV_32S)
    for i in range(1, retval):
        coords = np.where(labels == i)
        ymin = np.amin(coords[0])
        xmin = np.amin(coords[1])
        ymax = np.amax(coords[0])
        xmax = np.amax(coords[1])
        cells.append([ymin, xmin, ymax, xmax])

    cv2.imshow("test", labels)
    cv2.waitKey(-1)

    return cells