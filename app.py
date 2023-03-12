from flask import Flask, render_template, request
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Users/ADITYA/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'
import cv2
from pyaadhaar.utils import Qr_img_to_text, isSecureQr
import easyocr
import re
from PIL import Image
from pyaadhaar.utils import Qr_img_to_text, isSecureQr
from pyaadhaar.decode import AadhaarSecureQr
from pyaadhaar.decode import AadhaarOldQr
import xml.etree.ElementTree as ET
import numpy as np
import os

app = Flask(__name__)

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])

name = None
yob = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/aadhar", methods=['GET', 'POST'])
def aadhar():
    global name
    global yob
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

        cv2.imwrite('static/images/aadhar.png', img[y-20:y+h+20, x-20:x+w+20])

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
                name = root.attrib['name'].lower()
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
    global name
    global yob
    if request.method == 'POST':
        # Get the uploaded file from the HTML form
        file = request.files['pan_img']
        # Read the image file using OpenCV
        img = cv2.imdecode(np.fromstring(
            file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        cv2.imwrite('static/images/pan_img.png', img)

        pan_img = 'pan_img.png'
        # # Save the uploaded image temporarily for debugging purposes
        dpi = 80
        fig_width, fig_height = int(img.shape[0]/dpi), int(img.shape[1]/dpi)
        mylst = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
        reader = easyocr.Reader(['en']) 
        result = reader.readtext(img) 

        for (bbox, text, prob) in result:
            if prob >= 0.5:
                # display 
                mylst.append(text)
                print(f'Detected text: {text} (Probability: {prob:.2f})')

                # get top-left and bottom-right bbox vertices
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

                # create a rectangle for bbox display
                cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=10)

                # put recognized text
                cv2.putText(img=img, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1)

        if len(mylst) == 0:
            context = None
        else:
            # Parse the XML document
            mylst_1 = []
            for i in mylst: 
                data = re.sub(r'[^a-zA-Z0-9\s]', ' ', i) 
                data = data.lower()  
                mylst_1.append(data)
            mylst_1[ : ] = [' '.join(mylst_1[ : ])]
            status = 'Reupload your correct PAN image' 
            if 'income' in mylst_1[0]:
                status = "Pan image uploaded successully" 
            if name in mylst_1[0] and yob in mylst_1[0]:
                verified = 'Pan card and Aaadhar card verified'
            else:
                verified = 'Re-upload Pan Card. Not Verified' 
            context = {'pan_card_status': status, 'verified_status':verified}

    else:
        context = None
        pan_img = None

    return render_template('pan.html', context=context, pan_img=pan_img)


if __name__ == '__main__':
    app.run(debug=True)
