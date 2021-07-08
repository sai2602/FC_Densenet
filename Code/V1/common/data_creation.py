import cv2 as cv
import numpy as np
import os, sys
from common import depthmap_helpers
from common.CsvReader import CsvReader
from common import pointcloud_helpers
from common import depthmap_helpers
from common.labelling_helpers import CalculateLabelByGrid, CalculateLabelsByDepthmap, ManualLabelling
from matplotlib.pyplot import imread

from common.data_augmentation import DataAugmentation
import glob
import math
import random

def resize_img_with_padding(img, resize, padding):
    h , w , c = img.shape
    resize_img = np.ones(resize)*padding
    if h < resize[0] and w > resize[1]:
        crop_pad = int((w - resize[1])/2)
        img1 = img[:,crop_pad:w-crop_pad,:]
        add_pad = int((resize[0]-h)/2)
        resize_img[add_pad:resize[0]-add_pad,:,:] = img1

    return resize_img


def create_depth_csv_contours(img_path, labels_path, save_path_contours, save_path_csv, num_files):
    temp = np.zeros((800, 600))
    for i in range(1,num_files+1):
        print('working on', img_path + str(i) + '.png')
        img = cv.imread(img_path + str(i) + '.png', cv.IMREAD_GRAYSCALE)
        img_cropped= img[140:940, 660:1260]
        if np.array_equal(img_cropped,temp):
            temp = img_cropped
            continue
        else:
            print('working on', labels_path + str(i) + '.png')
            seg_mask = cv.imread(labels_path + str(i) + '.png', cv.IMREAD_GRAYSCALE)
            seg_mask_cropped = seg_mask[140:940, 660:1260]
            CsvReader.save(img_cropped,seg_mask_cropped,save_path_csv + str(i) + '.csv')
            contours = depthmap_helpers.apply_contour_filter(seg_mask_cropped,simulataion_data=True)
            cv.imwrite(save_path_contours + str(i) + '.png', contours)
            temp = img_cropped

def write_cropped_pointclouds(data_path, lower_limit_x = 100000, upper_limit_x=-100000,
                              lower_limit_y=100000, upper_limit_y=-100000,
                              lower_limit_z=100000, upper_limit_z=-100000):
    for file in glob.glob(data_path + '/*/*/*'):
        if file.endswith('scan.xyz'):
            print('Cropping point cloud of', file)
            point_cloud = pointcloud_helpers.load_xyz(file)
            point_cloud_cropped = pointcloud_helpers.crop_point_cloud(point_cloud,lower_limit_x,upper_limit_x,
                                                                      lower_limit_y,upper_limit_y,
                                                                      lower_limit_z,upper_limit_z)
            labels = np.zeros((point_cloud_cropped.shape[0],1))
            pointcloud_helpers.write_xyz(file[:-4] + '-cropped.xyz',point_cloud_cropped)

def write_instance_labels_point_cloud(data_path, grid_cell_size=0.5, max_distance_euclidean=0.5):
    LabelCalculator = CalculateLabelByGrid(grid_cell_size=grid_cell_size, empty_dirs=data_path + 'bin/',
                                           max_distance_euclidean=max_distance_euclidean)

    for i in range(1, 11):
        for file in glob.glob(data_path + str(i) + '/*/*'):
            if file.endswith('scan-cropped.xyz'):
                print('Working on', file)
                point_cloud = pointcloud_helpers.load_xyz(file)
                labels = LabelCalculator.calculate_instance_labels(point_cloud, i)
                CsvReader.saveDataRowByRow(point_cloud, labels, file[:-4] + '-labelled-gcz_' + str(int(grid_cell_size*10))
                                           + '-mde_' + str(int(max_distance_euclidean*10)) + '.xyz')

def write_projected_depthmaps(data_path, resolution, img_size = None, z_flip = True ):
    count = 0
    for file in glob.glob(data_path + '/*/*/*'):
        if file.endswith('scan-cropped.xyz'):
            print('Projecting depthmap of', file)
            point_cloud = pointcloud_helpers.load_xyz(file)
            if count ==0:
                input_min = np.min(point_cloud, 0)
                input_max = np.max(point_cloud, 0)
                input_span = input_max - input_min
                image_width = math.ceil(float(input_span[0]) / float(resolution))
                image_height = math.ceil(float(input_span[1]) / float(resolution))
                img_size = (image_width,image_height)
                count+=1
            depthmap, labels = pointcloud_helpers.project_to_depth_image_orthogonal_by_resultion(point_cloud, resolution,z_flip=z_flip, img_size=img_size)
            CsvReader.save(depthmap,labels,file.rsplit('/',1)[0] + '/depthmap_orthorgonal_r1_w' + str(img_size[0]) + '_h'+ str(img_size[1]) + '.csv')
    file_name = 'depthmap_orthorgonal_r1_w' + str(img_size[0]) + '_h'+ str(img_size[1]) + '.csv'
    return file_name

def calculateLabels(data_path, file_name, max_distance_euclidean, region_x1 = 45, region_x2 = 550,region_y1 = 95,region_y2 = 340, max_z = 240, min_z = 10):
    label_calculator = CalculateLabelsByDepthmap(data_path + 'bin', file_name=file_name, max_distance_euclidean=max_distance_euclidean)
    file_labels = file_name
    for i in range(1, 11):
        for file in glob.glob(data_path + str(i) + '/*/*'):
            if file.endswith(file_name):
                print('Calculating Labels of', file)
                depthmap, labels = CsvReader.load(file)
                labels = label_calculator.get_instance_labels(depthmap, i, region_x1, region_x2, region_y1, region_y2, max_z , min_z)
                CsvReader.save(depthmap, labels, file[:-4] + '-labelled-filtered.csv')
                file_labels = file.rsplit('/', 1)[1][:-4] + '-labelled-filtered.csv'
    return file_labels

def create_real_depthmap_contours(data_path, file_name,):
    for file in glob.glob(data_path + '/*'):
        if file.endswith(file_name):
            print('Creating Contours of', file)
            depthmap, labels = CsvReader.load(file)
            contours = depthmap_helpers.apply_contour_filter(labels)
            #cv.imwrite(file.rsplit('/',1)[0] + '/contour' + file.rsplit('/',1)[1][8:][:-4] + '.png', contours)
            cv.imwrite(file.rsplit('/', 1)[0] + '/contour'  + '.png', contours)
            cv.imwrite(file.rsplit('/',1)[0] + '/depthmap' +  '.png',depthmap.astype(np.uint8))

def start_manual_labelling(data_path, file_name,):
    for file in glob.glob(data_path +  '/*'):
        if file.endswith(file_name):
            print('Working on', file)
            manual_labeller = ManualLabelling(file,save_label_rgb=True)
            manual_labeller.create_label_image(save_label_rgb=True)
            manual_labeller.do_manual_labelling()

def create_data_augmentations(data_path, file_name):
    for file in glob.glob(data_path + '/*'):
        if file.endswith(file_name):
            print('Creating data augmentation of', file)
            depthmap, labels = CsvReader.load(file)
            aug = DataAugmentation(depthmap,labels,file,img_size=1000)
            # aug.flip_lr()
            # aug.flip_up()
            # aug.rotate90()
            # aug.transpose()
            # # aug.translate(direction='right',shift=10)
            # # aug.translate(direction='left',shift=10)
            # # aug.translate(direction='up',shift=10)
            # # aug.translate(direction='down',shift=10)
            # aug.rotate(angle=2,rotation='x')
            # aug.rotate(angle=-2, rotation='x')
            # aug.rotate(angle=2, rotation='y')
            # aug.rotate(angle=-2, rotation='y')
            # aug.rotate(angle=30, rotation='z')
            # aug.rotate(angle=45, rotation='z')
            # aug.rotate(angle=60, rotation='z')
            # aug.rotate(angle=90, rotation='z')
            # aug.rotate(angle=120, rotation='z')
            aug.rotate(angle=135, rotation='z')
            # aug.rotate(angle=150, rotation='z')
            # aug.rotate(angle=180, rotation='z')
            break

def create_data(data_path_real,filename_real,data_path_simulated,filename_simulated, save_path):
    count = 0
    for file in glob.glob(data_path_real + '/*'):
        if file.endswith(filename_real):
            print('Creating data augmentation of', file)
            depthmap, labels = CsvReader.load(file)
            aug = DataAugmentation(depthmap,labels,save_path,img_size=1000,simulation_data=False)
            aug.copy_original(file_name= str(count+1))
            aug.flip_lr(file_name= str(count+2))
            aug.flip_up(file_name= str(count+3))
            aug.rotate90_1(file_name= str(count+4))
            aug.rotate90_3(file_name= str(count+5))
            aug.transpose(file_name= str(count++6))
            aug.rotate(file_name= str(count+7),angle=2,rotation='x')
            aug.rotate(file_name= str(count+8),angle=-2, rotation='x')
            aug.rotate(file_name= str(count+9),angle=2, rotation='y')
            aug.rotate(file_name= str(count+10),angle=-2, rotation='y')
            aug.rotate(file_name= str(count+11),angle=30, rotation='z')
            aug.rotate(file_name= str(count+12),angle=45, rotation='z')
            aug.rotate(file_name= str(count+13),angle=60, rotation='z')
            aug.rotate(file_name= str(count+14),angle=90, rotation='z')
            aug.rotate(file_name= str(count+15),angle=120, rotation='z')
            aug.rotate(file_name= str(count+16),angle=135, rotation='z')
            aug.rotate(file_name= str(count+17),angle=150, rotation='z')
            aug.rotate(file_name= str(count+18),angle=180, rotation='z')
            count+=18

    for file in glob.glob(data_path_simulated + '/*'):
        if file.endswith(filename_simulated):
            print('Creating data augmentation of', file)
            depthmap, labels = CsvReader.load(file)
            aug = DataAugmentation(depthmap,labels,save_path,img_size=1000,simulation_data=True)
            aug.copy_original(file_name= str(count+1))
            aug.flip_lr(file_name= str(count+2))
            aug.flip_up(file_name= str(count+3))
            aug.rotate90_1(file_name= str(count+4))
            aug.rotate90_3(file_name= str(count+5))
            aug.transpose(file_name= str(count++6))
            aug.rotate(file_name= str(count+7),angle=2,rotation='x')
            aug.rotate(file_name= str(count+8),angle=-2, rotation='x')
            aug.rotate(file_name= str(count+9),angle=2, rotation='y')
            aug.rotate(file_name= str(count+10),angle=-2, rotation='y')
            aug.rotate(file_name= str(count+11),angle=30, rotation='z')
            aug.rotate(file_name= str(count+12),angle=45, rotation='z')
            aug.rotate(file_name= str(count+13),angle=60, rotation='z')
            aug.rotate(file_name= str(count+14),angle=90, rotation='z')
            aug.rotate(file_name= str(count+15),angle=120, rotation='z')
            aug.rotate(file_name= str(count+16),angle=135, rotation='z')
            aug.rotate(file_name= str(count+17),angle=150, rotation='z')
            aug.rotate(file_name= str(count+18),angle=180, rotation='z')
            count+=18

def create_train_test_list(data_path):
    trainFile = data_path +  "Train_test_list/train_list.txt"
    testFile = data_path +  "Train_test_list/test_list.txt"

    f_train = open(trainFile, "w+")
    f_test = open(testFile, "w+")
    files = []
    count = 0
    for file in glob.glob(data_path + 'DepthMapsCSV/*'):
        if file.endswith('.csv'):
            files.append(file)
            count+=1
    num_Test_files = int(0.05*count)
    random.shuffle(files)

    count = 0
    for file in files:
        if count <= num_Test_files:
            f_test.write(file)
            f_test.write('\n')
            count+=1
        else:
            f_train.write(file)
            f_train.write('\n')
            count+=1

    f_train.close()
    f_test.close()

def prepare_data_with_contours(data_path):

    for i in range(1,4897):
        sfile = data_path.rsplit('/',2)
        file_depth = data_path + '/' + str(i) + '.csv'
        print('Working on file', file_depth)
        file_contour = sfile[0] + '/Contours/' + str(i) + '.png'
        #file_depth = sfile[0] + '/' + 'DepthMapsPNG_16bit' + '/' + str(i) + '.png'
        #depthmap = cv.imread(file_depth, cv.IMREAD_ANYDEPTH)
        #df = pd.read_csv(file_depth)
        #depthmap = df.values
        depthmap, l = CsvReader.load(file_depth)
        #labels_org = cv.imread(file_contour,cv.IMREAD_GRAYSCALE)
        labels_org = imread(file_contour)
        labels1 = np.where(labels_org == 255, 1, labels_org)
        labels = np.where(depthmap == 0, 0, labels1)
        file_save  = sfile[0] + '/DepthMapsCSV_Contours/' + str(i) + '.csv'
        CsvReader.save(depthmap,labels,file_save)