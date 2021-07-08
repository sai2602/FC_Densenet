import tensorflow as tf
import cv2 as cv
import numpy as np
def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    combined = tf.concat(axis=2, values=[image, label]) 
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))    
    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    return img_crop, label_crop  

def random_crop_and_pad_image(image, crop_h, crop_w, ignore_label=255):
    image_shape = tf.shape(image)
    pad = tf.image.pad_to_bounding_box(image, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))    
    last_image_dim = tf.shape(image)[-1]
    img_crop = tf.random_crop(pad, [crop_h,crop_w,3])
    img_crop.set_shape((crop_h, crop_w, 3))
    return img_crop

def testSegmentationMask(image_path = '', segmentation_path = '', save_path = '',  value=255):

    img_mask = cv.imread(segmentation_path,cv.IMREAD_GRAYSCALE)
    img = cv.imread(image_path,cv.IMREAD_GRAYSCALE)

    img_one_id = np.where(img_mask == value, img_mask, img_mask*0)
    print('image_one_id')
    cv.imshow('Image_id',img_one_id)
    cv.waitKey(0)
    if save_path!='':
        cv.imwrite(save_path,img_mask)


def testCustomFilter(image_path = '', save_path='', kernel_size = 3):
    img = cv.imread(image_path)
    kernel = np.ones((kernel_size,kernel_size),dtype=np.float32)
    kernel /= (kernel_size*kernel_size)

    dst = cv.filter2D(img,-1,kernel)
    if save_path != '':
        cv.imwrite(save_path, dst)
    cv.imshow('Normalized_box_filter',dst)
    cv.waitKey(0)

def getCannyEdges(image_path = '', save_path=''):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img, 50, 255)
    cv.imwrite(save_path, edges)