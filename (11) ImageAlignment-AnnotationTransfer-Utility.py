### "ImageAlignment-AnnotationTransfer-Utility.py"

#Code for Option 1:
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load points from JSON files
def load_points(json_file):
    with open(json_file) as f:
        data = json.load(f)
    points = []
    for shape in data['shapes']:
        points.append(shape['points'][0])
    return np.array(points, dtype=np.float32)

# Paths to the JSON files and images
json_file_large = '/Users/gustavszviedris/Downloads/LineUpTest/T51_SC.json'
json_file_small = '/Users/gustavszviedris/Downloads/LineUpTest/T51_HM.json'
image_large_path = '/Users/gustavszviedris/Downloads/LineUpTest/T51.jpg'
image_small_path = '/Users/gustavszviedris/Downloads/LineUpTest/T51.png'

# Load points
points_large = load_points(json_file_large)
points_small = load_points(json_file_small)

# Load images
image_large = cv2.imread(image_large_path)
image_small = cv2.imread(image_small_path)

# Calculate scaling factor and translation
def similarity_transform(src_points, dst_points):
		#Calculate the center (centroid) of the points
    src_center = np.mean(src_points, axis=0)
    dst_center = np.mean(dst_points, axis=0)
    
    #Demean the points by subtracting the center
    src_demean = src_points - src_center
    dst_demean = dst_points - dst_center
    
    #Calculate the norms (lengths) of the demeaned points
    norm_src = np.linalg.norm(src_demean)
    norm_dst = np.linalg.norm(dst_demean)
    
    #Calculate the scaling factor
    scale = norm_dst / norm_src
    
    #Normalize the demeaned points
    src_demean /= norm_src
    dst_demean /= norm_dst
    
    #Calculate the rotation matrix
    A = np.dot(dst_demean.T, src_demean)
    U, S, Vt = np.linalg.svd(A)
    R = np.dot(U, Vt)
    
    #Construct the transformation matrix
    M = np.eye(3)
    M[0:2, 0:2] = scale * R
    M[0:2, 2] = dst_center.T - scale * np.dot(R, src_center.T)
    
    return M

# Compute the transformation matrix
M = similarity_transform(points_large, points_small)

# Warp the large image to align with the small image
height, width, channels = image_small.shape
aligned_image_large = cv2.warpAffine(image_large, M[0:2], (width, height))

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title("Small Image")
plt.imshow(cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title("Aligned Large Image")
plt.imshow(cv2.cvtColor(aligned_image_large, cv2.COLOR_BGR2RGB))
plt.show()

cv2.imwrite('aligned_large_image3.jpg', aligned_image_large)
