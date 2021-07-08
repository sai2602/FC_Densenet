import numpy as np


class NpzReader:
    NPZ_FILE_ENDING = ".csv"

    @classmethod
    def is_npz(cls, file):
        return file.endswith(cls.NPZ_FILE_ENDING)

    @staticmethod
    def save(depth_data, label_data, path):
        np.savez(path, depth_data, label_data)

    @staticmethod
    def load(path):
        result = np.load(path)
        return result["arr_0"], result["arr_1"]