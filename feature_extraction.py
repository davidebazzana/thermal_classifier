import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, canny
from skimage.filters import sobel, threshold_otsu
from skimage.filters.rank import entropy
from skimage.morphology import disk, ellipse
from skimage.util import img_as_ubyte
from scipy.stats import skew, kurtosis
import scipy
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import math


def extract_features(image, bounding_box=None):
    if bounding_box is not None:
        x, y, w, h = bounding_box
        image = image[y:y+h, x:x+w]

    glcm = graycomatrix(image.astype(np.uint8), distances=[1, 2, 3, 5, 10], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean(axis=1)
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean(axis=1)
    homogeneity = graycoprops(glcm, 'homogeneity').mean(axis=1)
    energy = graycoprops(glcm, 'energy').mean(axis=1)
    correlation = graycoprops(glcm, 'correlation').mean(axis=1)
    entropy = graycoprops(glcm, 'entropy').mean(axis=1)
    
    features = {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
        "entropy": entropy,
    }

    return features


def compute_image_entropy_mask(image):
    # Load image and convert to grayscale
    # image = Image.open(image_path).convert('L')
    # gray = np.array(image)

    # gray_uint8 = img_as_ubyte(image)

    selem = ellipse(10, 5)    
    entropy_img = entropy(image, selem)

    thresh = threshold_otsu(entropy_img)
    mask = entropy_img > thresh

    return mask


def get_bounding_box(mask):
    # Convert boolean mask to uint8 for OpenCV
    binary_uint8 = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        bounding_box = cv2.boundingRect(largest_contour)
        return bounding_box

    return None


def show_image(image):
    cv2.imshow("", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_image(image, bounding_box=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    if bounding_box is not None:
        rect = patches.Rectangle(
            (bounding_box[0], bounding_box[1]), bounding_box[2], bounding_box[3],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
    ax.set_title("Image with Bounding Boxes")
    ax.axis('off')
    plt.show()


def plot_image_and_entropy(image, entropy_img, max_entropy):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Image")

    axes[0].scatter(max_entropy[1], max_entropy[0], color='blue', marker='+', s=100)

    axes[1].hist(entropy_img.ravel(), bins=256)
    axes[1].set_title('Entropy Histogram')
    axes[1].set_xlabel('Entropy Value')
    axes[1].set_ylabel('Pixel Count')
    
    plt.show()
    
