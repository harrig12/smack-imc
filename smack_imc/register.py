# registration_helpers.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform 
from skimage.color import rgb2gray
from skimage.filters import gaussian
from scipy.stats.mstats import winsorize
from skimage.io import imread, imsave, imshow


def make_gray(channel):
    channel = channel.copy()
    out = channel * (255/channel.max())
    out = out.astype('uint8')
    return out

def preprocess_IMC_nuclear(image, channel_axis=2, arcsinh_normalize=True, arcsinh_cofactor=5, winsorize_limits=[None,0.2], binarization_threshold=2, sigma=1):
    image = image.copy()
    if arcsinh_normalize: 
        image = np.arcsinh(image/arcsinh_cofactor)
    image = winsorize(image, limits = winsorize_limits, axis=channel_axis)
    image = gaussian(image, sigma=sigma, channel_axis=channel_axis)
    image = image.sum(channel_axis) > binarization_threshold
    image = make_gray(image)
    return image

def preprocess_mask_nuclear(image, sigma=1):
    image = image.copy()
    image = gaussian(image, sigma=sigma)
    image = image > 0
    image = make_gray(image)
    return image

def preprocess_IF_nuclear(image, binarization_threshold=0.1, sigma=1):
    image = image.copy()
    image = gaussian(image, sigma=sigma, channel_axis = 2)
    image = rgb2gray(image) 
    image = image > binarization_threshold
    image = make_gray(image)
    return image

def approx_scale(image, template, image_downscale_axis=0):
    image = image.copy()
    template = template.copy()
    scale_factor = (template.shape[image_downscale_axis])/(image.shape[image_downscale_axis])
    image = transform.rescale(image, scale_factor, anti_aliasing=False)
    image = make_gray(image)
    return image


def register(image, template, max_features, keep_percent):
    image = image.copy()
    template = template.copy()
    try:
        orb = cv2.ORB_create(max_features, fastThreshold=0, edgeThreshold=0)
        (kpsA, descsA) = orb.detectAndCompute(image, None)
        (kpsB, descsB) = orb.detectAndCompute(template, None)
        # match the features
        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(descsA, descsB, None)
        # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
        matches = sorted(matches, key=lambda x:x.distance)
        # keep only the top percentile matches
        keep = int(len(matches) * keep_percent)
        matches = matches[:keep]
        ptsA = np.array([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.array([kpsB[m.trainIdx].pt for m in matches])
        # register
        (h, w) = template.shape
        M, _mask = cv2.estimateAffine2D(ptsA, ptsB)
        aligned = cv2.warpAffine(image, M, (w,h))
        return aligned, M

    except cv2.error as e:
        print('registration failed')
        return np.zeros(template.shape), np.NaN


def score_registration(image, template, transformation_matrix):
    image = image.copy()
    template = template.copy()
    # Get aligned pixel selection
    sel = np.ones(image.shape)
    (h, w) = template.shape
    sel_aligned = cv2.warpAffine(sel, transformation_matrix, (w,h)) > 0
    # Drop pixels outside of selection
    image_aligned = cv2.warpAffine(image, transformation_matrix, (w,h))
    template[~sel_aligned]=0
    image_aligned[~sel_aligned]=0
    # Score agreement
    score = ((image_aligned & template)>0).sum() - np.logical_xor(image_aligned, template).sum() 
    # Percent of maximum 
    ppc = ((image_aligned & template)>0).sum() / (template>0).sum()
    return score, ppc

def plot_registration(registered_image, template, cmap1=5, cmap2=0):
    base = cv2.applyColorMap(template, cmap1)
    overlay = cv2.applyColorMap(registered_image, cmap2)
    comb = cv2.addWeighted(base, 0.5, overlay, 0.5, 0, overlay)
    return imshow(comb)