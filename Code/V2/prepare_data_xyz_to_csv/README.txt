1. Used to convert .xyz to the expected projection format as .csv
2. Standard co-ordinates file added in the folder
3. Format of co-ordinates in file: lower_x upper_x;lower_y upper_y;lower_z upper_z
4. If Z value is not as expected i.e., if Z value for top of the bin is NOT GREATER than Z value for bottom of the bin, then, set 'FLIP_XYZ_FILE' to True. This will set the Z axis to the expected type.
   If Z value at the top of the bin > Z value at bottom of the bin, then 'FLIP_XYZ_FILE' should be set to false.
5. in 'main function' (at line 83), change base_path to basepath of where the .xyz file is.
6. Due to the restriction in a module imported from prepare_data, please ensure that the file is 3 folders deep from base_path. So path of .xyz file will be base_path/folder_1/folder_2/folder_3/name_of_xyz_file.xyz
   This issue will be fixed in the next release.
7. In the function call 'crop_label_point_cloud' __main__ function, change the path of the co-ordinates.txt file in 'text_path'
8. Run the script and it will generate a 'projection.csv' file in the same folder as the .xyz file