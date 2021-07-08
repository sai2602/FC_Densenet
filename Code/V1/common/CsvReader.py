import numpy as np
import csv
import os


class CsvReader:

    CSV_FILE_ENDING = ".csv"

    @classmethod
    def is_csv(cls, file):
        return file.endswith(cls.CSV_FILE_ENDING)

    @staticmethod
    def load(file, funcLabelConverter = None, img_size=None):
        if not img_size:
            img_size = CsvReader.getImageSize(file)

        with open(file, newline='') as csvfile:
            has_header = csv.Sniffer().has_header(csvfile.read(1024))
            csvfile.seek(0)
            reader = csv.reader(csvfile)
            # Skip header
            if has_header:
                next(reader)

            # depthData = np.zeros(img_size, dtype=np.float32)
            # labelData = np.zeros(img_size, dtype = np.float32)
            # for row in reader:
            #     indices = np.array(row[0:2], dtype=np.int32)
            #     depthData[indices[0] - 1, indices[1] - 1] = np.float32(row[2])
            #     labelData[indices[0] - 1, indices[1] - 1] = funcLabelConverter(np.float32(row[3]))
            x = list(reader)
            result = np.array(x).astype(dtype=np.float32)

            depthData = result[:,2]
            depthData =  np.reshape(depthData,img_size)
            labelData = result[:,3]
            labelData = np.reshape(labelData,img_size)

            #return depthData
            return depthData, labelData

    @staticmethod
    def getImageSize(file):
        with open(file, newline='') as csvfile:
            has_header = csv.Sniffer().has_header(csvfile.read(1024))
            csvfile.seek(0)
            reader = csv.reader(csvfile)
            # Skip header
            if has_header:
                next(reader)
            # just read the file twice... first get dimensions of image -> improve and save it as header
            cur_width = 0
            cur_height = 0
            indexing_0 = False

            for row in reader:
                indices = np.array(row[0:2], dtype=np.int32)
                x = indices[0]
                y = indices[1]
                if [x,y] ==[0,0]:
                    indexing_0 = True
                if x > cur_width:
                    cur_width = x
                if y > cur_height:
                    cur_height = y
            if indexing_0:
                return [cur_width+1, cur_height+1]
            else:
                return [cur_width, cur_height]

    @staticmethod
    def save(depthData, labelData, filename):
        if os.path.exists(filename):
            os.remove(filename)

        with open(filename, 'w+') as file:
            for x in range(depthData.shape[0]):
                for y in range(depthData.shape[1]):
                    line = str(x) + "," + str(y) + "," + str(depthData[x, y]) + "," + str(labelData[x, y]) + "\n"
                    #line = str(x) + "," + str(y) + "," + str(depthData[x, y])  + "\n"
                    file.write(line)

    @staticmethod
    def saveDataRowByRow(data, labels = None, filename = None):
        if os.path.exists(filename):
            os.remove(filename)

        with open(filename, 'w+') as file:
            for index in range(data.shape[0]):
                row = data[index]
                if len(row) == 3:
                    file.write(str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "," + str(labels[index]) + "\n")
                else:
                    file.write(str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "," + str(row[3]) + "\n")


