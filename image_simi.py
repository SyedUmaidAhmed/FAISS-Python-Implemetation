import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def pad_descriptors(descriptors, target_length):
    pad_length = max(target_length - descriptors.shape[0], 0)
    padded_descriptors = np.pad(descriptors, ((0, pad_length), (0, 0)), mode='constant')
    return padded_descriptors


    

# Load images
image_a = cv2.imread("IMG_0002.JPG")
image_b = cv2.imread("IMG_0003.JPG")



# Create SIFT detector
sift = cv2.SIFT_create(nOctaveLayers=4, contrastThreshold=0.03, edgeThreshold=10)

# Compute SIFT keypoints and descriptors for the images
keypoints_a, descriptors_a = sift.detectAndCompute(image_a, None)
keypoints_b, descriptors_b = sift.detectAndCompute(image_b, None)

# Choose a target length for padding (equal to the maximum keypoint count)
target_length = max(descriptors_a.shape[0], descriptors_b.shape[0])

# Pad the descriptors arrays
padded_descriptors_a = pad_descriptors(descriptors_a, target_length)
padded_descriptors_b = pad_descriptors(descriptors_b, target_length)

# Calculate cosine similarity using sklearn's cosine_similarity function
similarity = cosine_similarity(padded_descriptors_a, padded_descriptors_b)

print(f"Cosine similarity between padded descriptor arrays: {similarity[0][0]:.6f}")
