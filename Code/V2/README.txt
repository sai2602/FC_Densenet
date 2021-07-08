To train: (Run main.py after the following changes)

in main.py file,
1. Change mode to 'train'
2. Change train_data to base path of where the training data text file is.
3. Change val_data to base path of where the test data text file is.
4. Change ckpt to where the check point has to be saved.

in Model_FCD.py file,
1. Create 2 text files containing paths to training data and test data respectively. (Currently supports .npy and .png files)
2. 'train_image' should be the name of text file containing training data paths and 'val_image' should be name of the text file containing test data paths.


To test: (Run main.py after the following changes)

As a first step, run the script available in 'prepare_data_xyz_to_csv' folder (Contains a README as well for use)

in main.py file,
1. Change mode to 'infer'
2. Change ckpt to where the check point has to be restored from.

in Model_FCD.py file,
1. Create a text file containing the complete paths to the csv files on which the predicition should be carried out on (These csv files are created by the script in 'prepare_data_xyz_to_csv' folder).
2. 'inference_file_path' should contain the complete path of the text file containing inferernce paths of .csv files