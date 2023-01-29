import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform 
from skimage.color import rgb2gray
from skimage.filters import gaussian
from scipy.stats.mstats import winsorize
from skimage.io import imread, imsave, imshow


def make_gray(image):
    """
    Make an image grayscale

    Args:
        iamge (numpy.ndarray): The image channel to convert to grayscale.

    Returns:
        (numpy.ndarray): The grayscale version of the input image.
    """
    channel = channel.copy()
    out = channel * (255/channel.max())
    out = out.astype('uint8')
    return out

def preprocess_IMC_nuclear(image, channel_axis=2, arcsinh_normalize=True, arcsinh_cofactor=5, winsorize_limits=[None,0.2], binarization_threshold=2, sigma=1):
    """
    Perform all preprocessing steps to sum across nuclear channels in the image. 

    Args:
        image (numpy.ndarray): The image to be preprocessed.
        channel_axis (int, optional): The axis along which the channels are stored, by default 2.
        arcsinh_normalize (bool, optional): Whether to apply arcsinh normalization, by default True.
        arcsinh_cofactor (int, optional): The cofactor to use for arcsinh normalization, by default 5.
        winsorize_limits (list, optional): The lower and upper limits to use for winsorizing the image, by default [None, 0.2].
        binarization_threshold (int, optional): The threshold to use for binarizing the image, by default 2.
        sigma (int, optional): The sigma value to use for gaussian blur filtering the image, by default 1.

    Returns:
        numpy.ndarray : The preprocessed image.
    """
    image = image.copy()
    if arcsinh_normalize: 
        image = np.arcsinh(image/arcsinh_cofactor)
    image = winsorize(image, limits = winsorize_limits, axis=channel_axis)
    image = gaussian(image, sigma=sigma, channel_axis=channel_axis)
    image = image.sum(channel_axis) > binarization_threshold
    image = make_gray(image)
    return image

def preprocess_mask_nuclear(image, sigma=1):
    """
    Perform all preprocessing steps to the nuclear mask. 

    Args:
        image (numpy.ndarray): The image to be preprocessed.
        sigma (int, optional): The sigma value to use for gaussian blur filtering the image, by default 1.

    Returns:
        numpy.ndarray : The preprocessed image.
    """
    image = image.copy()
    image = gaussian(image, sigma=sigma)
    image = image > 0
    image = make_gray(image)
    return image

def preprocess_IF_nuclear(image, binarization_threshold=0.1, sigma=1):
    """
    Perform all preprocessing steps to sum across nuclear channels in the image. 

    Args:
        image (numpy.ndarray): The image to be preprocessed.
        binarization_threshold (int, optional): The threshold to use for binarizing the image, by default 0.1.
        sigma (int, optional): The sigma value to use for gaussian blur filtering the image, by default 1.

    Returns:
        numpy.ndarray : The preprocessed image.
    """
    image = image.copy()
    image = gaussian(image, sigma=sigma, channel_axis = 2)
    image = rgb2gray(image) 
    image = image > binarization_threshold
    image = make_gray(image)
    return image

def approx_scale(image, template, image_downscale_axis=0):
    """
    This function approximately re-scales an image by comparing it to a template image.
    
    Parameters:
        image (np.ndarray): The input image to be scaled.
        template (np.ndarray): The template image to compare to.
        image_downscale_axis (int): The reference axis along which the image will be downscaled according to (default 0).
    
    Returns:
        np.ndarray : The tansformed image.
    """
    image = image.copy()
    template = template.copy()
    scale_factor = (template.shape[image_downscale_axis])/(image.shape[image_downscale_axis])
    image = transform.rescale(image, scale_factor, anti_aliasing=False)
    image = make_gray(image)
    return image


def register(image, template, max_features, keep_percent):
    """
    This function registers an image to a template image by finding corresponding points between the two images and warping the input image to align with the template.
    
    Parameters:
        image (np.ndarray): The input image to be registered.
        template (np.ndarray): The template image to register to.
        max_features (int): The maximum number of features to extract from the image and template for registration.
        keep_percent (float): The percentage of features to keep after filtering (0-1).
    
    Returns:
        np.ndarray: The registered image.
    """
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
    """
    This function scores the registration of an image to a template image by comparing the registered image to the template.
    
    Parameters:
        image (np.ndarray): The input image that has been registered.
        template (np.ndarray): The template image that the input image has been registered to.
        transformation_matrix (np.ndarray): The transformation matrix used for registration.
    
    Returns:
        (int, float): A tuple containing two metrics for registration quality. The first metric is the "Overlap Score" and represents the number of agreeing pixels, minus the number of disagreeing. It may be positive or negative. The second is the the "Percent Pixels Correct" which is calculated as the number of pixels matched, out of the total number which could be expected given the transformation matrix. This may be between 0 and 1, where 1 represents a perfect registration. In some cases of a poor registration, the Percent Pixels Correct may be 1 despite having a low the Overlap Score.
    """
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
    """This function plots the registration of an image with a template.

    Parameters:
        registered_image (ndarray): The image that has been registered with the template.
        template (ndarray): The template image used for registration.
        cmap1 (int, optional): The `opencv colormap <https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html/>`_ to use for the registered image. Default is 5.
        cmap2 (int, optional): The `opencv colormap <https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html/>`_ to use for the template image. Default is 0.

    Returns:
        matplotlib.image.AxesImage
    """
    base = cv2.applyColorMap(template, cmap1)
    overlay = cv2.applyColorMap(registered_image, cmap2)
    comb = cv2.addWeighted(base, 0.5, overlay, 0.5, 0, overlay)
    return imshow(comb)