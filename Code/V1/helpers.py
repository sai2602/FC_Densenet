import numpy as np
import cv2 as cv


def read_train_test_list(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    data = [x.strip("\n") for x in content]

    return data


def stack_labels(label):
    labels = []
    batch_size = label.shape[0]
    base_mask = label[0, :, :, 0]
    channel_1 = np.where(base_mask == 1, 0, 1)
    base_stack = np.dstack((channel_1, base_mask))
    labels.append(base_stack)

    for index in range(1, batch_size):
        append_mask = label[index, :, :, 0]
        channel_1 = np.where(append_mask == 1, 0, 1)
        append_stack = np.dstack((channel_1, append_mask))
        labels.append(append_stack)
        if index % 100 == 0:
            print(index)
    print(len(labels))
    labels = np.array(labels, dtype=np.uint8)
    return labels


class BatchDatasetReader:
    def __init__(self, image_list, mask_list, image_size=(800, 800)):
        self.image_files = image_list
        self.image_size = image_size
        self.mask_files = mask_list
        self.batch_offset = 0
        self.epochs_completed = 0
        self.images = []
        self.annotations = []
        self.epoch_finished = False

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.image_files):
            # Finished epoch
            self.epochs_completed += 1
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        return self.read_images(start, end)

    def read_images(self, start, end):
        depth_image = []
        annotation = []
        for i in range(start, end):
            image_file = self.image_files[i]
            mask_file = self.mask_files[i]
            depthmap = cv.imread(image_file.strip("\r"))[:, :, 0]
            labels_org = cv.imread(mask_file.strip("\r"))[:, :, 0]
            labels1 = np.where(labels_org == 21, 1, labels_org)
            labels = np.where(labels1 == 13, 0, labels1)
            depth_image.append(depthmap)
            annotation.append(labels)
        depth_image = np.array(np.expand_dims(depth_image, 3))
        annotation = np.array(np.expand_dims(annotation, 3))

        self.images = depth_image
        self.annotations = annotation

        return self.images, self.annotations
