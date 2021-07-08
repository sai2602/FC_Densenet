"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
from common.CsvReader import CsvReader
import cv2 as cv
#from matplotlib.pyplot import imread

def read_train_test_list(data_dir, prefix):

    with open(data_dir + '/' +prefix,'r') as file:
        content = file.readlines()

    records = [x.strip('\n') for x in content]

    return records


class BatchDatasetReader:
    def __init__(self, records_list, image_size = (1000,1000)):
        self.files = records_list
        self.image_size = image_size
        self.image_list = []
        self.images = []
        self.annotations = []
        self.read_list()
        #self.shuffle_list()
        self.batch_offset = 0
        self.epochs_completed = 0
        self.epoch_finished = False

    def shuffle_list(self):
        print('Shuffling list')
        np.random.shuffle(self.image_list)

    def read_list(self):
        print("Reading image list")
        file_list = []
        for i in range(len(self.files)):
            file_list.append(self.files[i])
        self.image_list = np.array(file_list)

    def next_batch(self,batch_size, annotation_name = 'Contours', annotation_suffix = '.png', use_CSV = True):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset >= len(self.image_list):
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            #perm = np.arange(self.images.shape[0])
            #np.random.shuffle(perm)
            #self.images = self.images[perm]
            #self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        return self.read_images(start,end, annotation_name, annotation_suffix, use_CSV)

    def read_images(self,start,end, annotation_name, annotation_suffix, use_CSV):
        depth_image = []
        annotation = []
        for i in range(start,end):
            file = self.image_list[i]
            #file = self.files[i]
            if use_CSV:
                depthmap, labels = CsvReader.load(file,img_size=[self.image_size[0],self.image_size[1]])
            else:
                # sfile = file.rsplit('/', 2)
                # file_depth = sfile[0] + '/' + 'DepthMapsPNG_16bit' + '/' + sfile[2][:-4] + '.png'
                # #depthmap = cv.imread(file_depth, cv.IMREAD_ANYDEPTH)
                # file_depth = file
                depthmap = cv.imread(file)[:,:,0]
                labels = np.zeros(depthmap.shape)

            if not annotation_name == "":
                sfile = file.rsplit('/',2)
                file_labels = sfile[0] + '/' + annotation_name + '/' + sfile[2][:-4] + annotation_suffix
                #labels_org = cv.imread(file_labels,cv.IMREAD_GRAYSCALE)
                labels_org = cv.imread(file_labels)[:,:,0]
                labels1 = np.where(labels_org == 21, 1, labels_org)
                labels = np.where(labels1 == 13, 0, labels1)
            depth_image.append(depthmap)
            annotation.append(labels)
        depth_image = np.array(np.expand_dims(depth_image, 3))
        annotation = np.array(np.expand_dims(annotation, 3))

        self.images = depth_image
        self.annotations = annotation

        return self.images, self.annotations, self.epoch_finished

def normalize_meanstd(a, axis=None):
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std



