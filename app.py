from flask import Flask, render_template, request
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Users/ADITYA/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'
import cv2
from pyaadhaar.utils import Qr_img_to_text, isSecureQr
from cv2 import *
from PIL import Image
from pyaadhaar.utils import Qr_img_to_text, isSecureQr
from pyaadhaar.decode import AadhaarSecureQr
from pyaadhaar.decode import AadhaarOldQr
import xml.etree.ElementTree as ET
import numpy as np
import os

app = Flask(__name__)

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign')
def sign():
    return render_template('sign.html')


@app.route("/aadhar", methods=['GET', 'POST'])
def aadhar():
    if request.method == 'POST':
        # Get the uploaded file from the HTML form
        file = request.files['aadhar_img']
        # Read the image file using OpenCV
        img = cv2.imdecode(np.fromstring(
            file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # # Save the uploaded image temporarily for debugging purposes

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, thresh = cv2.threshold(
            gray, 1, 10, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Find contours in the image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the largest contour (assumed to be the signature)
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the signature from the image
        x, y, w, h = cv2.boundingRect(largest_contour)

        cv2.imwrite('static/images/aadhar.png', img[y-10:y+h+10, x-10:x+w+10])

        img_name = 'aadhar.png'
        # Extract Aadhaar card details from the image
        qrData = Qr_img_to_text('static/images/aadhar.png')
        if len(qrData) == 0:
            context = None
        else:
            # Parse the XML document
            for i in qrData:
                root = ET.fromstring(i)
                uid = root.attrib['uid']
                name = root.attrib['name']
                gender = root.attrib['gender']
                yob = root.attrib['yob']
                context = {"UID": uid, "Name": name,
                           "Gender": gender, "YOB": yob}
    else:
        context = None
        img_name = None

    return render_template('aadhar.html', context=context, img_name=img_name)


@app.route('/pan', methods=['GET', 'POST'])
def pan():
    if request.method == 'POST':

        file = request.files['aadhar_img'] 

        img = cv2.imdecode(np.fromstring(
            file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)

        text1 = pytesseract.image_to_data(threshed,output_type='data.frame')

        text2 = pytesseract.image_to_string(threshed, lang="ind")

        text = text1[text1.conf != -1]

        lines = text.groupby('block_num')['text'].apply(list)

        mylst = [] 
        for i in lines:
            mylst.extend(i)

        if(len(mylst)==0): 
            context = None 
        else: 
            pnr = ""
            name = ""
            yob = ""
            if 'Number' in mylst: 
                ind = mylst.index('Number') 
                pnr += mylst[ind + 1]
            if 'Name' in mylst: 
                ind = mylst.index('Name') 
                name = name + mylst[ind + 1] + " " 
                name = name + mylst[ind + 2] + " " 
            if 'Birth' in mylst: 
                ind = mylst.index('Birth') 
                yob = yob + mylst[ind + 1]
            context = {"PNR": pnr, "Name": name, "YOB": yob}
    else: 
        context = None


        return render_template('pan.html', context=context)

@app.route('/capture_image', methods=['GET', 'POST'])
def capture_img():
        return render_template('capture_image.html')


if __name__ == '__main__':
    app.run(debug=True)
