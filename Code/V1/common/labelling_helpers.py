from common import pointcloud_helpers
import numpy as np
import time
from common.CsvReader import CsvReader
from abc import ABC, abstractmethod
import glob
from common.depthmap_helpers import *
from scipy import spatial
import cv2 as cv

'''
    Script for calculating real labels. There's a couple of parameters to choose for this script:
        'filled_dirs': path where the cycle data of point clouds with filled boxes are
        'empty_dirs': path where the cycle data of point clouds with empty boxes are
        'xy_grid_cell_size': Cell-edge-length for xy-grid in mm. If this parameter is chosen bigger a wider area is used
            for "searching near neighbours". The algorithm is putting all points of the empty box into this xy_grid.
            Later it searches  for neighbours of points in all neighbouring grid_cells.
        'max_distance_euclidean': Maximum euclidean distance to consider to points to be "neighbours". If a point of
            a filled point cloud has no "neighbour" in the empty point clouds / grid -> then it's considered as
            workpiece. Otherwise it's considered as not-workpiece
'''


# class AbstractLabelCalculator(ABC):
#
#     @abstractmethod
#     def calculate_labels(self, point_cloud):
#         pass

class CalculateLabelByGrid:

    xy_grid = None
    xy_grid_cell_size = None
    max_distance_euclidean = None

    def __init__(self, grid_cell_size, empty_dirs, max_distance_euclidean):
        #self.xy_grid = {}
        self.grids = []
        self.xy_grid_cell_size = grid_cell_size
        self.max_distance_euclidean = max_distance_euclidean

        # 1. Read all empty datas and put them into the grid
        for file in glob.glob(empty_dirs+"/*/scan-cropped.xyz"):
            print("Working on", file)
            point_cloud = pointcloud_helpers.load_xyz(file)
            self.grids.append(self.add_grid(point_cloud))
            print("added 'empty point cloud' " + str(file) + "'to grid points")


    def add_grid(self, point_cloud):
        xy_grid = {}
        for p in point_cloud:
            x_grid_index_offset = int(p[0] / self.xy_grid_cell_size)
            y_grid_index_offset = int(p[1] / self.xy_grid_cell_size)

            if x_grid_index_offset in xy_grid:
                if y_grid_index_offset in xy_grid[x_grid_index_offset]:
                    xy_grid[x_grid_index_offset][y_grid_index_offset].append(p)
                else:
                    xy_grid[x_grid_index_offset][y_grid_index_offset] = [p]
            else:
                xy_grid[x_grid_index_offset] = {y_grid_index_offset: [p]}
        return xy_grid


    def calculate_semantic_labels(self, point_cloud, label):
        labels = np.zeros([point_cloud.shape[0]])
        nNeighbourBags = 1
        counter = -1
        ts_start = time.time()
        for p_filled in point_cloud:
            cur_time = time.time() - ts_start
            if counter % 10000 == 0:
                print(
                    '[GridBased] Done with ' + str(counter) + ' points from total n=' + str(point_cloud.shape[0]) + ' in ' +
                    str(cur_time) + 's')
            counter += 1

            # First get the neighbouring grid_cells
            x_grid_index_offset = int(p_filled[0] / self.xy_grid_cell_size)
            for i_x_grid in range(nNeighbourBags * 2):
                x_grid_index_cur = x_grid_index_offset + i_x_grid - nNeighbourBags
                #if x_grid_index_cur in self.xy_grid:
                y_grid_index_offset = int(p_filled[1] / self.xy_grid_cell_size)
                for i_y_grid in range(nNeighbourBags * 2):
                    y_grid_index_cur = y_grid_index_offset + i_y_grid - nNeighbourBags
                    #if y_grid_index_cur in self.xy_grid[x_grid_index_cur]:
                    # Now look for all points in the grid cell if they are considered as "neighbour"
                    count_label = -1
                    for i in range(len(self.grids)):
                        if self.findInGrid(p_filled,i, x_grid_index_cur, y_grid_index_cur):
                            count_label+=1
                    labels[counter] = label + count_label
        return labels

    def findInGrid(self, p_filled, grid_num, x_grid_index_cur, y_grid_index_cur ):
        if x_grid_index_cur not in self.grids[grid_num]:
            return True
        if y_grid_index_cur not in self.grids[grid_num][x_grid_index_cur]:
            return True
        for p_empty in self.grids[grid_num][x_grid_index_cur][y_grid_index_cur]:
            difference = p_filled - p_empty
            euclidean_distance = np.sqrt(np.dot(difference, difference))
            if euclidean_distance > self.max_distance_euclidean:
                return True


    def calculate_instance_labels(self, point_cloud, label):
        labels = self.calculate_semantic_labels(point_cloud,label)
        self.grids.append(self.add_grid(point_cloud))
        return labels

class CalculateLabelsByDepthmap:
    def __init__(self, empty_dirs_path, file_name,  max_distance_euclidean):
        self.path = empty_dirs_path.rsplit('\\',1)[0] + '\\'
        self.max_distance = max_distance_euclidean
        self.file_name = file_name
        self.create_height_table_with_labels(empty_dirs_path,label=0)

    def create_height_table_with_labels(self,depthmap_path, label):
        for file in glob.glob(depthmap_path + '\\*\\*'):
            if file.endswith(self.file_name):
                print('Working on', file)
                depthmap, labels = CsvReader.load(file)
                self.height_table = np.zeros(depthmap.shape)
                self.label_table = np.zeros(depthmap.shape)
                for x in range(depthmap.shape[0]):
                    for y in range(depthmap.shape[1]):
                        self.height_table[x,y] = depthmap[x,y]
                        self.label_table[x,y] = label

    def get_instance_labels(self,depthmap, label, region_x1 = 45, region_x2 = 550,region_y1 = 95,region_y2 = 340, max_z = 240, min_z = 10):
        labels_template = np.zeros(depthmap.shape)
        labels = self.label_table
        for x in range( depthmap.shape[0]):
            for y in range(depthmap.shape[1]):
                value_curr = depthmap[x, y]
                if value_curr > max_z or value_curr < min_z:
                    labels[x, y] = 0
                    continue
                value_table = self.height_table[x , y ]
                difference = np.array([x, y, value_curr]) - np.array([x, y, value_table])
                euclidean_distance = np.sqrt(np.dot(difference, difference))
                if euclidean_distance > self.max_distance:
                    labels[x, y] = label
                else:
                    labels[x, y] = self.label_table[x, y]
        labels_template[region_x1:region_x2,region_y1:region_y2] = labels[region_x1:region_x2,region_y1:region_y2]
        labels = apply_label_filter(labels_template)

        self.label_table = labels
        self.height_table = depthmap
        return labels

class CalculateLabelByGridXYZ:

    xy_grid = None
    xy_grid_cell_size = None


    def __init__(self, grid_cell_size, empty_dirs):
        self.xyz_grid = []
        self.xyz_grid_cell_size = grid_cell_size
        ts_start = time.time()
        # 1. Read all empty datas and put them into the grid
        for file in glob.glob(empty_dirs+"/*/scan.xyz"):
            print("Working on", file)
            point_cloud_empty_path = file
            point_cloud_empty = pointcloud_helpers.load_xyz(point_cloud_empty_path)
            point_cloud_empty = pointcloud_helpers.crop_point_cloud(point_cloud_empty, upper_limit=-410,
                                                                      lower_limit=-660)
            grid_min = np.min(point_cloud_empty,0)
            grid_max = np.max(point_cloud_empty,0)
            counter = 0
            for p in point_cloud_empty:
                x_grid_index_offset = int(p[0] / grid_cell_size)
                y_grid_index_offset = int(p[1] / grid_cell_size)
                z_grid_index_offset = int(p[2] / grid_cell_size)
                grid_coord = [x_grid_index_offset, y_grid_index_offset, z_grid_index_offset]
                # if grid_coord in self.xyz_grid:
                #     continue
                # else:
                self.xyz_grid.append(grid_coord)

                if counter % 10000 == 0:
                    cur_time = time.time() - ts_start
                    print(
                        '[GridBased] Done with ' + str(counter) + ' points from total n=' + str(
                            point_cloud_empty.shape[0]) + ' in ' +
                        str(cur_time) + 's')
                counter+=1
            print("added 'empty point cloud' " + str(point_cloud_empty_path) + "'to grid points with shape")

    def calculate_labels(self, point_cloud):
        labels = np.zeros([point_cloud.shape[0]])
        counter = -1
        ts_start = time.time()
        for p_filled in point_cloud:
            cur_time = time.time() - ts_start
            if counter % 10000 == 0:
                print(
                    '[GridBased] Done with ' + str(counter) + ' points from total n=' + str(point_cloud.shape[0]) + ' in ' +
                    str(cur_time) + 's')
            counter += 1
            x_grid_index_offset = int(p_filled[0] / self.xyz_grid_cell_size)
            y_grid_index_offset = int(p_filled[1] / self.xyz_grid_cell_size)
            z_grid_index_offset = int(p_filled[2] / self.xyz_grid_cell_size)
            grid_coord = [x_grid_index_offset, y_grid_index_offset, z_grid_index_offset]
            if grid_coord in self.xyz_grid:
                labels[counter] = 0
            else:
                labels[counter] = 1

        return labels


class RegionBasedCalculator:

    def __init__(self, corners):
        self.corners = corners

    def get_axis_and_dot_product_range(self, i0, i1):
        axis = self.corners[i0]-self.corners[i1]
        d0 = np.dot(axis, self.corners[i0])
        d1 = np.dot(axis, self.corners[i1])

        if d1 < d0:
            return axis, d1, d0
        return axis, d0, d1

    def calculate_labels(self, point_cloud):
        labels = np.ones([point_cloud.shape[0]])
        n_dims  = point_cloud.shape[1]
        counter = -1
        ts_start = time.time()

        indices_to_check = np.where(
            np.sum(
                np.logical_and(
                    point_cloud < np.max(self.corners, 0),
                    point_cloud > np.min(self.corners, 0)), 1) == n_dims)

        dir_x_axis, dx0, dx1 = self.get_axis_and_dot_product_range(0, 1)
        dir_y_axis, dy0, dy1 = self.get_axis_and_dot_product_range(0, 3)
        dir_z_axis, dz0, dz1 = self.get_axis_and_dot_product_range(0, 4)
        dir_sx_axis, dsx0, dsx1 = self.get_axis_and_dot_product_range(2, 3)
        dir_sy_axis, dsy0, dsy1 = self.get_axis_and_dot_product_range(2, 1)

        max = np.array([0.0, 0.0, 0.0])
        min = np.array([100000.0, 100000.0, 100000.0])

        for ind in range(point_cloud.shape[0]):
            dx = np.dot(dir_x_axis, point_cloud[ind])
            dy = np.dot(dir_y_axis, point_cloud[ind])
            dz = np.dot(dir_z_axis, point_cloud[ind])
            dsx = np.dot(dir_sx_axis,point_cloud[ind])
            dsy = np.dot(dir_sy_axis, point_cloud[ind])

            # if dx >= dx0 and dx <= dx1 and dy >= dy0 and dy <= dy1 and dz >= dz0 and dz <= dz1:
            #     if dsx < dsx0 or dsx > dsx1:
            #         print("Effected by Diagonal Points: ", point_cloud[ind])
            #if dx >= dx0 and dx <= dx1 and dy >= dy0 and dy <= dy1 and dz >= dz0 and dz <= dz1 and dsx >= dsx0 and dsx <= dsx1 and dsy >= dsy0 and dsy <= dsy1 :
            if dx >= dx0 and dx <= dx1 and dy >= dy0 and dy <= dy1 and dz >= dz0 and dz <= dz1:
                labels[ind] = 0

                for i in range(3):
                    if point_cloud[ind][i] > max[i]:
                        max[i] = point_cloud[ind][i]
                    if point_cloud[ind][i] < min[i]:
                        min[i] = point_cloud[ind][i]

            if counter % 10000 == 0:
                cur_time = time.time() - ts_start
                print(
                    '[RBC] Done with ' + str(ind) + ' points from total n=' + str(len(indices_to_check[0]))+
                    ' and total=' + str(point_cloud.shape[0]) + ' in ' + str(cur_time) + 's')
            counter += 1

        return labels

    def calculate_labels_using_planes(self,point_cloud,data):

        ts_start = time.time()
        vecOX1=data[0]-data[1]
        vecOZ1=data[0]-data[4]
        vecOX2=data[3]-data[2]
        vecOZ2=data[3]-data[7]

        vec_OY1=data[1]-data[2]
        vec_OZ1=data[1]-data[5]
        vec_OY2=data[0]-data[3]
        vec_OZ2=data[0]-data[4]

        vec__OX1 = data[0]-data[1]
        vec__OY1 = data[0]-data[3]
        vec__OX2 = data[4]-data[5]
        vec__OY2 = data[4]-data[7]

        normal_xz1 = np.cross(vecOX1,vecOZ1)
        normal_xz2 = np.cross(vecOX2,vecOZ2)
        normal_yz1 = np.cross(vec_OY1,vec_OZ1)
        normal_yz2 = np.cross(vec_OY2,vec_OZ2)
        normal_xy1 = np.cross(vec__OX1,vec__OY1)
        normal_xy2 = np.cross(vec__OX2,vec__OY2)
        normal_xy1 = np.array([-0.00308109, -0.00146115, 0.999994])
        #labels = np.ones([point_cloud.shape[0]])
        labels = np.ones(len(point_cloud))
        counter = 0
        #for ind in range(point_cloud.shape[0]):
        for ind in range(len(point_cloud)):
            vecXZ1 = data[0] - point_cloud[ind]
            vecXZ2 = data[3] - point_cloud[ind]
            vecYZ1 = data[1] - point_cloud[ind]
            vecYZ2 = data[0] - point_cloud[ind]
            vecXY1 = data[0] - point_cloud[ind]
            #vecXY1 = np.array([-168.946442, 579.212280, -649])-point_cloud[ind]
            #vecXY1 = np.array([-168.430466, 577.632141, -649]) - point_cloud[ind]
            vecXY1 = np.array([-144.094589, 586.242188, -653.8]) - point_cloud[ind]
            #vecXY1 = np.array([-169.178528, 580.185364, -648.3]) - point_cloud[ind]
            vecXY2 = data[4] - point_cloud[ind]

            valXZ1 = np.dot(normal_xz1,vecXZ1)
            valXZ2 = np.dot(normal_xz2,vecXZ2)
            valYZ1 = np.dot(normal_yz1,vecYZ1)
            valYZ2 = np.dot(normal_yz2,vecYZ2)
            valXY1 = np.dot(normal_xy1,vecXY1)
            valXY2 = np.dot(normal_xy2,vecXY2)

            #if valXZ1 < 0 and valXZ2 > 0 and valYZ1 < 0 and valYZ2 > 0 and valXY1 > 0:
                #labels[ind]=0
            if valXY1 < 0:
                labels[ind]=0

            if counter % 10000 == 0:
                cur_time = time.time() - ts_start
                #print(
                    #'[RBC] Done with ' + str(ind) + ' points from total n=' + str(counter)+
                    #' and total=' + str(point_cloud.shape[0]) + ' in ' + str(cur_time) + 's')
                print(
                    '[RBC] Done with ' + str(ind) + ' points from total n=' + str(counter) +
                    ' and total=' + str(len(point_cloud)) + ' in ' + str(cur_time) + 's')
            counter+=1

        return labels

class ManualLabelling:
    def __init__(self, file_path, save_label_rgb = True):
        self.depthmap, self.labels = CsvReader.load(file_path)
        self.path = file_path.rsplit('\\',1)[0]
        self.rbg_value = {
            0 : (0,0,0),
            1 : (255,0,0),
            2 : (0,255,0),
            3 : (0,0,255),
            4 : (255,255,0),
            5: (0, 255, 255),
            6: (255, 0, 255),
            7: (128, 0, 0),
            8: (0, 128, 0),
            9: (0, 0, 128),
            10: (128, 128, 0),
            11: (0, 128, 128),
            12: (128, 0, 128)
        }
    def create_label_image(self, save_label_rgb = True):
        self.image = np.zeros((self.labels.shape[0], self.labels.shape[1] , 3), np.uint8)
        for x in range(self.image.shape[0]):
            for y in range(self.image.shape[1]):
                value = self.labels[x,y]
                self.image[x,y] = self.rbg_value[int(value)]
        if save_label_rgb:
            cv.imwrite(self.path + '\\label_rgb.png', self.image)

    def left_click_and_drag(self, event, x, y, flags, param):
        global refPt, cropping

        if event == cv.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = False
            print(refPt)

        elif event == cv.EVENT_LBUTTONUP:
            refPt.append((x, y))
            cropping = False
            cv.rectangle(self.image, refPt[0], refPt[1], (0, 255, 0), 1)
            cv.imshow("image", self.image)

    def do_manual_labelling(self):
        global refPt
        #image = self.image
        labels = self.labels
        clone = self.image.copy()
        orgclone = self.image.copy()
        undoclone = self.image.copy()

        orgLables = labels.copy()
        undolabels = labels.copy()
        cv.namedWindow("image")
        cv.setMouseCallback("image", self.left_click_and_drag)

        while True:
            cv.imshow("image", self.image)
            key = cv.waitKey(0)

            if key == ord("r"):  ##reset
                image = orgclone.copy()
                clone = orgclone.copy()
                labels = orgLables

            elif key==ord("p"):
                print('Color picker')
                x_start = min(x[0] for x in refPt)
                y_start = min(x[1] for x in refPt)
                self.selected_label = labels[y_start,x_start]
                self.selcted_color = self.rbg_value[int(labels[y_start,x_start])]
                print(refPt)
                print(labels[y_start,x_start])
                print(self.rbg_value[int(labels[y_start,x_start])])

            elif key == ord("c"):  ##Copy selected area and work
                print('Label correcting')
                undoclone = clone.copy()
                undolabels = labels.copy()
                if len(refPt) == 2:
                    x_start = min(x[0] for x in refPt)
                    x_end = max(x[0] for x in refPt)
                    y_start = min(x[1] for x in refPt)
                    y_end = max(x[1] for x in refPt)
                    print("Image shape :", clone.shape, "xStart:", x_start, "xEnd:", x_end, "yStart:", y_start, "yEnd:",
                          y_end)
                    if x_start < 0:
                        x_start = 0
                    if y_start < 0:
                        y_start = 0
                    if x_end > clone.shape[1]:
                        x_end = clone.shape[1]
                    if y_end > clone.shape[0]:
                        y_end = clone.shape[0]

                    for x in range(x_start, x_end):
                        for y in range(y_start, y_end):
                            # print(labels[y,x])
                            if self.selected_label!=0:
                                if labels[y,x] !=0 and labels[y,x]!= self.selected_label:
                                    labels[y,x] = self.selected_label
                                    clone[y,x] = self.selcted_color
                            else:
                                if labels[y,x]!= self.selected_label:
                                    labels[y,x] = self.selected_label
                                    clone[y,x] = self.selcted_color

                    # roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                    # cv2.imshow("ROI", clone)
                    # cv2.waitKey(100)
                    refPt = []
                    self.image = clone.copy()

            elif key == ord("u"):  ##undo previously selected region
                clone = undoclone.copy()
                labels = undolabels.copy()
                self.image = undoclone.copy()

            elif key == ord("x"):  ###undo selected region
                self.image = clone.copy()

            elif key == ord("q"):  ####Quit
                break

            elif key == ord("s"):  ####Save and Quit
                print("Saving: ", self.path +  "\\depthmap_manual_filtered1.csv")
                cv.imwrite(self.path +  "\\labels_manual_filtered1.png", clone)
                CsvReader.save(self.depthmap, labels, self.path +  "\\depthmap_manual_filtered1.csv")
                break

    cv.destroyAllWindows()












