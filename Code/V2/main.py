import argparse
from Model_FCD import DenseTiramisu
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="train")
parser.add_argument("--train_data", default="/data/Packaging_detection/Raw/",
                    help="Directory for training images")
parser.add_argument("--val_data", default="/data/Packaging_detection/Raw/",
                    help="Directory for validation images")
parser.add_argument("--ckpt", default="/data/Packaging_detection/Raw/Model_Logs_packaging/model.ckpt-26",
                    help="Directory for storing model checkpoints")
parser.add_argument("--layers_per_block", default="2,5,8,11,15,16",
                    help="Number of layers in dense blocks")
parser.add_argument("--batch_size", default=1,
                    help="Batch size for use in training", type=int)
parser.add_argument("--epochs", default=100,
                    help="Number of epochs for training", type=int)
parser.add_argument("--num_threads", default=1,
                    help="Number of threads to use for data input pipeline", type=int)
parser.add_argument("--growth_k", default=7, help="Growth rate for Tiramisu", type=int)
parser.add_argument("--num_classes",   default=2, help="Number of classes", type=int)
parser.add_argument("--learning_rate", default=1e-4,
                    help="Learning rate for optimizer", type=float)
parser.add_argument("--infer_data", default="data/infer")
parser.add_argument("--output_folder", default="/data/Packaging_detection/Raw/Train_Set_CSV/")


def main():
    FLAGS = parser.parse_args()
    layers_per_block = [int(x) for x in FLAGS.layers_per_block.split(",")]

    tiramisu = DenseTiramisu(FLAGS.growth_k, layers_per_block, FLAGS.num_classes)

    if FLAGS.mode == 'train':
        tiramisu.train(FLAGS.train_data, FLAGS.val_data, FLAGS.ckpt,
                       FLAGS.batch_size, FLAGS.epochs, FLAGS.learning_rate)
    elif FLAGS.mode == 'infer':
        tiramisu.infer(FLAGS.train_data, FLAGS.batch_size, FLAGS.ckpt, FLAGS.output_folder)


if __name__ == "__main__":
    main()
