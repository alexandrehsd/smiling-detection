from PIL import Image
import os

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