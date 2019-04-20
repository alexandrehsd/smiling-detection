'''
    This script is meant for separate the labeled images from
    the non-labeled images. The labeled images are in the 
    ~/.../lfwcrop_grey/faces/ folder while the labeled images
    are being stored in the ~/.../lfwcrop_grey/labeled_faces/ .

    The labels are in the NON-SMILE.txt and in the SMILE.txt files.
    Basically, all peoples that are not smiling has its names' first 
    letter from A to J, while the smiling people has its names's 
    first letter between J and Z.

    This code uses the cp command to copy the labeled filed from
    the source directory to the destination directory.
'''

import os
from PIL import Image


# --------------------------------------------------
# Copying the labeled samples from the src_folder to 
# the dest_folder 
# --------------------------------------------------

# Note: src_folder and dest_folder are absolute paths

src_folder = '/home/alexandre/Documentos/Git/Smiling-Detection/lfwcrop_color/faces/{}'
dest_folder = '/home/alexandre/Documentos/Git/Smiling-Detection/lfwcrop_color/labeled_faces/{}'

with open('./lfwcrop_color/NON-SMILE_list.txt', 'r') as non_smile:

    # splitlines() returns a list, where the first element is the file name
    line = non_smile.readline().splitlines()[0]

    while line:
        os.system('cp ' + src_folder.format(line) + ' ' + dest_folder.format(line))
        line = non_smile.readline().splitlines()

        if line == []:
            break

        line = line[0]

with open('./lfwcrop_color/SMILE_list.txt', 'r') as smile:

    line = smile.readline().splitlines()[0]

    while line:
        os.system('cp ' + src_folder.format(line) + ' ' + dest_folder.format(line))
        line = smile.readline().splitlines()

        if line == []:
            break

        line = line[0]

# --------------------------------------------------
# Converts .ppm images to .jpg images in dest_folder
# lastly, delete all .ppm images to save disk space
# --------------------------------------------------

os.chdir('./lfwcrop_color/labeled_faces/')

with open('../LABELED_DATA_list.txt', 'r') as data:

    line = data.readline().split('.')[0]
    while line:

        im = Image.open(line + '.ppm' )
        im.save(line + '.jpg')

        line = data.readline().split('.')

        if line == []:
            break

        line = line[0]

os.system('rm *.ppm')