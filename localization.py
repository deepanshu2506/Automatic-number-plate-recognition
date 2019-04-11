# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 16:52:20 2019

@author: pc
"""

from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
from pytesseract import image_to_string
import cv2
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib
n=0

for i in range(1,317):

    car_image = imread("001/testing"+str(i)+".jpg", as_grey=True)


    gray_car_image = car_image * 255
   # fig, (ax1, ax2) = plt.subplots(1, 2)
   # ax1.imshow(gray_car_image, cmap="gray")
    threshold_value = threshold_otsu(gray_car_image)
    binary_car_image = gray_car_image > threshold_value
    #ax2.imshow(binary_car_image)
    #plt.show()



    label_image = measure.label(binary_car_image)

    plate_dimensions = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions
    plate_objects_cordinates = []
    plate_like_objects = []
#fig, (ax1) = plt.subplots(1)
#ax1.imshow(gray_car_image, cmap="gray");
    plate_image = ""

    for region in regionprops(label_image):
        if region.area < 50:
        
            continue

        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col
 
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            plate_like_objects.append(binary_car_image[min_row:max_row,
                                  min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col,
                                              max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
        #ax1.add_patch(rectBorder)
            plate_image = car_image[min_row : max_row, min_col:max_col] 
        #plt.imshow(plate_image)
        
        
        

    
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
#clean_plate, rect = cleanPlate(plate_image)
#plate_im = Image.fromarray(clean_plate)
#plate_im = Image.open("001/testing5.jpg")
#text = image_to_string(plate_im, lang='eng')
#print ("Detected Text : ",text)
#plt.show()







    license_plates =[]
    for obj in plate_like_objects:    
        license_plates.append(np.invert(obj))

    characters = []  
    column_list = []
    for license_plate in license_plates:
        labelled_plate = measure.label(license_plate)

    #fig, ax1 = plt.subplots(1)
    #ax1.imshow(license_plate, cmap="gray")

        character_dimensions = (0.35*license_plate.shape[0], 0.60*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
        min_height, max_height, min_width, max_width = character_dimensions

        counter=0
    
        for regions in regionprops(labelled_plate):
            y0, x0, y1, x1 = regions.bbox
            region_height = y1 - y0
            region_width = x1 - x0
            
            if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
                roi = license_plate[y0:y1, x0:x1]
            

                rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                            linewidth=2, fill=False)
            #ax1.add_patch(rect_border)
           # 

                resized_char = resize(roi, (20, 20))
                characters.append(resized_char)
            
            
                column_list.append(x0)

#plt.show()




# load the model
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
    model = joblib.load(model_dir)

    classification_result = []
    for each_character in characters:
    # converts it to a 1D array
        each_character = each_character.reshape(1, -1);
        result = model.predict(each_character)
        classification_result.append(result)



    plate_string = ''
    for eachPredict in classification_result:
        plate_string += eachPredict[0]


# it's possible the characters are wrongly arranged
# since that's a possibility, the column_list will be
# used to sort the letters in the right order

    column_list_copy = column_list[:]
    column_list.sort()
    rightplate_string = ''
    for each in column_list:
        rightplate_string += plate_string[column_list_copy.index(each)]

    
    if rightplate_string !='':
        n=n+1
        print(rightplate_string)

print("number = " +str(n))
