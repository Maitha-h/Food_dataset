import json
from os.path import join as opj
from PIL import Image
import numpy


with open('C:\\Users\\Maith\\Documents\\Datasets\\Dataset_food-101\\meta\\labels.txt') as l:
    labels = l.read().splitlines()
    print(len(labels))


with open('C:\\Users\\Maith\\Documents\\Datasets\\Dataset_food-101\\meta\\test.json') as json_file:
    data = json.load(json_file)
    print(len(data))
    # print(data.keys())
    print(list(data[list(data.keys())[0]])[0])
    print(list(data[list(data.keys())[0]])[0].split("/")[1])
    imgclass = list(data[list(data.keys())[0]])[0].split("/")[0]
    imgnumber = list(data[list(data.keys())[0]])[0].split("/")[1]
    imgnumber = "%s%s%s%s" % (imgclass, '\\',imgnumber,'.jpg')
    img = opj('C:\\Users\\Maith\\Documents\\Datasets\\Dataset_food-101\\images', imgnumber)
    myimg = Image.open(img)
    label = numpy.asarray(imgclass)
    print("label",label)
    I = numpy.asarray(myimg)
    x_test = I.astype('float32') / 255.0
    print(x_test)

