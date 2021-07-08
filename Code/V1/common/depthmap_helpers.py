import numpy as np
import h5py
import scipy
import math
import random
import skimage.color
import skimage.io
import skimage.transform
def add_boarders(image, labels, patch_size, masked_value):
    """
            Adds a boarder to the image according to the array according to the patch_size
            The order will be evenly split to all directions

            :param image: (w_image x h_image) np array, usually a depth_map
            :param labels: (w_image x h_image) np array, labels for each depth image pixel
            :param patch_size: 2d vector describing the patch size in [w_patch,h_patch]
            :param masked_value: All boarder pixel are set to this value

            :returns ( w_image + w_patch , h_image + h_patch ) np array
    """
    result_shape = [image.shape[0] + patch_size[0], image.shape[1] + patch_size[1]]
    result = np.ones(result_shape) * masked_value
    labelsFinal = np.zeros(result_shape)
    xHalf = int(patch_size[0] / 2)
    yHalf = int(patch_size[1] / 2)
    result[xHalf:(result_shape[0] - xHalf -1), yHalf:(result_shape[1] - yHalf -1)] = image
    labelsFinal[xHalf:(result_shape[0] - xHalf - 1), yHalf:(result_shape[1] - yHalf - 1)] = labels
    return result, labelsFinal


def rescale_image_in_z(image, z_scaling_range, z_flip = False):
    # Rescale z
    z_min = np.min(np.min(image))
    z_max = np.max(np.max(image))
    input_z_scale = np.float32(z_max - z_min)
    output_z_range =np.float32(z_scaling_range[1] - z_scaling_range[0])
    #depthmap_centered = np.array(image)
    z_min_image = np.ones(image.shape)*z_min
    input_z_scale_image = np.ones(image.shape)*input_z_scale
    output_z_range_image = np.ones(image.shape) * output_z_range
    depthmap_centered = np.float32(image - z_min_image)
    if z_flip:
        image=input_z_scale_image-depthmap_centered

    result = np.float32(depthmap_centered * (output_z_range / input_z_scale) + z_scaling_range[0])

    print("rescaled image from ( " + str(z_min) + " - " + str(z_max) +
          " ) to ( " + str( np.min(np.min(result))) + " - " + str( np.max(np.max(result))) + " )")
    return result.astype(np.float32)

def apply_filter(image, masked_value, kernel_size):

    image_size = image.shape
    result = np.array(image)
    kernelDelta = int(kernel_size/2)


    for x in range(kernelDelta, image_size[0] - kernelDelta):
        for y in range(kernelDelta, image_size[1] - kernelDelta):

            if image[x][y] == masked_value:
                for i in range(-kernelDelta, kernelDelta+1):
                    for j in range(-kernelDelta, kernelDelta + 1):
                        result[x+i][y+j] = masked_value

    return result

def apply_contour_filter(image, masked_value =255, kernel_size = 3, simulataion_data = False):
    image_size = image.shape
    result = np.zeros(image_size)
    kernel = int(kernel_size/2)

    for x in range(kernel, image_size[0] - kernel):
        for y in range(kernel, image_size[1] - kernel):

            curr_value = image[x][y]
            if curr_value == 0 or curr_value == 13:
                if simulataion_data:
                    continue # for bin in simulation data
            count = 0
            for i in range(-kernel, kernel+1):
                for j in range(-kernel, kernel + 1):
                    if image[x+i][y+j] !=curr_value:
                        count+=1
                        break
            if count!=0:
                result[x][y] = masked_value
    return result

def apply_label_filter(labels, kernel_size = 3):
    image_size = labels.shape
    result = labels
    kernel = int(kernel_size/2)
    for x in range(kernel, image_size[0] - kernel):
        for y in range(kernel, image_size[1] - kernel):

            curr_value = labels[x,y]
            if curr_value == 0:
                continue
            count = 0
            labels_in_kernel = []
            for i in range(-kernel, kernel+1):
                for j in range(-kernel, kernel + 1):
                    if [i,j] == [0,0]:
                        continue
                    labels_in_kernel.append(labels[x+i,y+j])
                    if labels[x+i,y+j]==curr_value:
                        count+=1
            if count<=2:
                counts = np.bincount(labels_in_kernel)
                result[x,y] = np.argmax(counts)
    return result






def apply_dilation_filter_on_labels(image, labels,  masked_value =255, kernel_size = 3):
    image_size = image.shape
    result = labels
    kernel = int(kernel_size/2)
    for x in range(kernel, image_size[0] - kernel):
        for y in range(kernel, image_size[1] - kernel):
            curr_value = labels[x,y]


    for x in range(kernel, image_size[0] - kernel):
        for y in range(kernel, image_size[1] - kernel):
            curr_value = labels[x,y]
            count = 0
            if curr_value == masked_value:
                for i in range(-kernel, kernel+1):
                    for j in range(-kernel, kernel + 1):
                        if labels[x+i][y+j] !=masked_value:
                            count+=1
                            break
                if count>=2:
                    result[x,y] = masked_value
    return result








def blur_depth_image(depth_image, kernel_size, masked_value):
    """
            Blurs an image according to the following algorithm:
                - A pixel is only blurred if it's equal to the masked_value
                - Then the average of all neighbours unequal to masked_value is used for pixel
                - If there are no neighbours or the pixel is not equal to the masked_value, nothing is changed
                - Borders are not considered

            :param depth_image: (w_image x h_image) np array
            :param kernel_size: scalar, should be odd
            :param masked_value: the value which should be

            :returns ( w_image + w_patch , h_image + h_patch ) np array
    """
    # Initialize image with masked value
    image_size = depth_image.shape
    result = np.ones(image_size) * masked_value
    kernelDelta = int(kernel_size/2)

    for x in range(kernelDelta, image_size[0] - kernelDelta):
        for y in range(kernelDelta, image_size[1] - kernelDelta):

            if not depth_image[x, y] == masked_value:
                result[x, y] = depth_image[x, y]
            else:

                # For each pixel iterate over the kernel
                n_neighbours = 0
                sum_neighbours = 0

                for dx in range(kernel_size):
                    for dy in range(kernel_size):
                        value = depth_image[x - kernelDelta + dx, y - kernelDelta + dy]

                        if not value == masked_value:
                            n_neighbours += 1
                            sum_neighbours += value

                if n_neighbours > 0:
                    result[x, y] = sum_neighbours / n_neighbours
                # Other case not needed since image is initialized with masked_value

    return result

def blur_depth_image_with_labels(depth_image, labels, kernel_size, masked_value):
    """
            Blurs an image according to the following algorithm:
                - A pixel is only blurred if it's equal to the masked_value
                - Then the average of all neighbours unequal to masked_value is used for pixel
                - If there are no neighbours or the pixel is not equal to the masked_value, nothing is changed
                - Lables of the hole pixel is changed based on the voting of the lables of the neighbours
                - Borders are not considered

            :param depth_image: (w_image x h_image) np array
            :param labels: (w_image x h_image) np array
            :param kernel_size: scalar, should be odd
            :param masked_value: the value which should be

            :returns ( w_image + w_patch , h_image + h_patch ) np array
    """
    # Initialize image with masked value
    image_size = depth_image.shape
    labels_size  = labels.shape
    result = np.ones(image_size) * masked_value
    kernelDelta = int(kernel_size/2)

    for x in range(kernelDelta, image_size[0] - kernelDelta):
        for y in range(kernelDelta, image_size[1] - kernelDelta):

            if not depth_image[x, y] == masked_value:
                result[x, y] = depth_image[x, y]
            else:

                # For each pixel iterate over the kernel
                n_neighbours = 0
                sum_neighbours = 0
                labels_in_kernel = []
                label1 = 0          # Kiste
                label0 = 0

                for dx in range(kernel_size):
                    for dy in range(kernel_size):
                        value = depth_image[x - kernelDelta + dx, y - kernelDelta + dy]
                        label_pixel = labels[x - kernelDelta + dx, y - kernelDelta + dy]


                        if not value == masked_value:
                            n_neighbours += 1
                            sum_neighbours += value
                            labels_in_kernel.append(label_pixel)
                            # if label_pixel==1.0:
                            #     label1+=1
                            # if  label_pixel==0.0:
                            #     label0+=1

                if n_neighbours > 0:
                    result[x, y] = sum_neighbours / n_neighbours
                    labels[x, y] = np.argmax(np.bincount(labels_in_kernel))
                    # if label0 >= label1:
                    #     labels[x,y]=0.0
                    # else:
                    #     labels[x,y]=1.0
                # Other case not needed since image is initialized with masked_value

    return result, labels

def remove_stray_labels(depth_image, labels, kernel_size, masked_value):
    image_size = depth_image.shape
    labels_size = labels.shape
    result = labels
    kernelDelta = int(kernel_size / 2)

    for x in range(kernelDelta, image_size[0] - kernelDelta):
        for y in range(kernelDelta, image_size[1] - kernelDelta):
            if not labels[x,y] == masked_value:
                continue
            else:
                n_neighbours = 0
                sum_neighbours = 0
                labels_neighbours = []
                distance_labels = []
                for dx in range(kernel_size):
                    for dy in range(kernel_size):
                        value = depth_image[x - kernelDelta + dx, y - kernelDelta + dy]
                        label = labels[x - kernelDelta + dx, y - kernelDelta + dy]
                        height_diff = abs(value-depth_image[x,y])
                        if (dx,dy) != (1,1) :
                            distance_labels.append(abs(value-depth_image[x,y]))
                            labels_neighbours.append(label)

                        if label == masked_value:
                            sum_neighbours+=1
                if sum_neighbours >=5:
                    continue
                else:
                    index = np.argmin(distance_labels)
                    result[x,y] = labels_neighbours[int(index)]
    return result




def normalizeDepthMap(depth_image, labels):

    depthMin = np.min(depth_image)
    depthMax = np.max(depth_image)
    depthSpan = depthMax -depthMin
    depthMean = np.mean(depth_image)
    #print("Depth Mean ", depthMean)
    depth_image_norm = np.zeros(depth_image.shape)

    for x in range(depth_image.shape[0]):
        for y in range(depth_image.shape[1]):
            #print(depth_image[x][y])
            depth_image[x][y] = 200 * (depth_image[x][y]-depthMin)/depthSpan

            #depth_image_norm[x][y] = (depth_image[x][y]-depthMean)/1

            #print(depth_image_norm[x][y], depth_image[x][y])
    print("depthMin = ", np.min(depth_image))
    print("depthMax = ", np.max(depth_image))
    # depthMean = np.mean(depth_image)
    # print("Depth Mean ", depthMean)
    # for x in range(depth_image.shape[0]):
    #     for y in range(depth_image.shape[1]):
    #         #print(depth_image[x][y])
    #         depth_image_norm[x][y] = (depth_image[x][y] - depthMean) / 1

    #print("Depth Minimum", depthMin)
    #print("Depth Maxmimum", depthMax)

    return depth_image, labels

def postFilter(depth_image, labels):
    print("Shape ", depth_image.shape)
    depthMax = np.max(depth_image)
    for x in range(depth_image.shape[0]-1):
        for y in range(depth_image.shape[1]-1):
            if depth_image[x][y] == depthMax:
                labels[x][y] = 0

    return depth_image, labels

def adjustBorders(depth_image, labels, probability, patch_size):
    labelsFinal = np.zeros(depth_image.shape)
    probabilityFinal = np.zeros(depth_image.shape)
    xHalf = int(patch_size[0] / 2)
    yHalf = int(patch_size[1] / 2)
    labelsFinal[xHalf:(labelsFinal.shape[0] - xHalf ), yHalf:(labelsFinal.shape[1] - yHalf)] = labels
    probabilityFinal[xHalf:(labelsFinal.shape[0] - xHalf), yHalf:(labelsFinal.shape[1] - yHalf)] = probability

    return depth_image, labelsFinal, probabilityFinal

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0)]
    cropping = [0,0]
    iScropping = False
    crop = None
    padding_constant = 0

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            print("Exceeds maximum dimension, needs cropping")
            mode = "crop_centre"
            #scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(h * scale), round(w * scale)),
            order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
        image = np.pad(image, padding, mode='constant', constant_values=padding_constant)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    elif mode == "crop_centre":
        iScropping = True
        padding = [(0, 0), (0, 0)]
        h, w = image.shape[:2]
        y_start = (h//2)-(max_dim//2)
        x_start = (w//2)-(max_dim//2)
        if y_start >=0 and x_start>=0:
            image = image[y_start:max_dim+y_start, x_start:max_dim+x_start]
        elif y_start < 0:
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = 0
            right_pad = 0
            padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
            image = np.pad(image, padding, mode='constant', constant_values=padding_constant)
            image = image[:, x_start:max_dim + x_start]
            cropping = [y_start, x_start]
        elif x_start < 0:
            top_pad = 0
            bottom_pad = 0
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
            image = np.pad(image, padding, mode='constant', constant_values=padding_constant)
            image = image[y_start:max_dim + y_start, :]
            cropping = [y_start, x_start]
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), padding , cropping, iScropping

def flip_depthimage(depthmap):
    input_min = np.min(depthmap)
    input_max = np.max(depthmap)
    input_span = input_max - input_min

    depthmap = input_span - depthmap

    return depthmap

def remap_labels_to_org_image(depthmap_org, labels_resize, padding, cropping, iScropping, resize_size = 800):
    print("Original Depthmap shape : ", depthmap_org.shape)
    if iScropping == False:
        h, w = labels_resize.shape[:2]
        labels_org = labels_resize[padding[0][0]:h - padding[0][1], padding[1][0]:w - padding[1][1]]
        if labels_org.size == 0:
            labels_org = labels_resize[padding[0][0]:h - padding[0][1],
                         padding[1][0]:w - padding[1][1]]
        print("Converted Labels Shape : ", labels_org.shape)

    else:
        if cropping[0] >= 0 and cropping[1] >= 0:
            h_org, w_org = depthmap_org.shape[:2]
            labels_org = np.zeros(depthmap_org.shape)
            labels_org[cropping[0]:cropping[0] + resize_size, cropping[1]:cropping[1] + resize_size] = labels_resize
            print("Converted Labels Shape : ", labels_org.shape)

        elif cropping[0] < 0:  # h < w
            h_org, w_org = depthmap_org.shape[:2]
            labels_interm = labels_resize[padding[0][0]:resize_size - padding[0][1], :]
            labels_org = np.zeros(depthmap_org.shape)
            labels_org[:, cropping[1]:cropping[1] + resize_size] = labels_interm
            print("Converted Labels Shape : ", labels_org.shape)
        elif cropping[1] < 0:  # h > w
            h_org, w_org = depthmap_org.shape[:2]
            labels_interm = labels_resize[:, padding[1][0]:resize_size - padding[1][1]]
            labels_org = np.zeros(depthmap_org.shape)
            labels_org[cropping[0]:cropping[0] + resize_size, :] = labels_interm
            print("Converted Labels Shape : ", labels_org.shape)

    return labels_org