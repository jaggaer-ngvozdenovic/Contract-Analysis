#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pytesseract
import re
import numpy as np
from pytesseract import Output
from transformers import pipeline
from pdf2image import convert_from_path
import os


# In[2]:


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[3]:


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# In[4]:


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# In[5]:


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# In[6]:


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# In[7]:


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# In[8]:


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# In[9]:


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# In[10]:


def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


# In[11]:


def convert_pdf(file_path):
    # convert pdf to multiple image
    images = convert_from_path(file_path)
    # save images to directory
    temp_images = []
    for i in range(len(images)):
        folder = file_path.split('/')[0]
        image_path = folder + '/' + folder.split('.')[0] + f'{i}.jpg'
        images[i].save(image_path, 'JPEG')
        temp_images.append(image_path)
    return temp_images


# In[12]:


question_answering = pipeline('question-answering')
custom_config = r'--oem 3 --psm 6'


# In[13]:


alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


# In[14]:


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


# In[15]:


def read_pdf(file_path):
    images_list = convert_pdf(file_path)
    text = ""
    for i in images_list:
        img = cv2.imread(i)
        text = text + " \n " + pytesseract.image_to_string(img, config=custom_config)
    return text


# In[16]:


def read_file(file_path):
    file = file_path.split('.')
    ext = file[-1]
    if ext == "pdf":
        text = read_pdf(file_path)
    elif ext in ["jpg", "jpeg", "png"]:
        img = cv2.imread(file_path)
        text = pytesseract.image_to_string(img, config=custom_config)
    elif ext in ["txt", "doc"]:
        with open(file_path) as f:
            text = f.read()
    else:
        text = "The file is not in required format!"
    return text






