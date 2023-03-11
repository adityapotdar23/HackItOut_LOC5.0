from flask import Flask, render_template, request
import cv2
from pyaadhaar.utils import Qr_img_to_text, isSecureQr
from pyaadhaar.decode import AadhaarSecureQr
from pyaadhaar.decode import AadhaarOldQr

import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

@app.route("/") 
def home(): 
    return render_template('aadhaar.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    # Get the uploaded file from the HTML form
    file = request.files['image']
    # Read the image file using OpenCV
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    print(img)
    plt.imshow(img)
    cv2.imwrite('image2.png', img)

    qrData = list(Qr_img_to_text('D:\ADITYA\Documents\LOC Hack_It_Out\image2.png'))
    print(type(qrData)) 
    print(qrData)
    if len(qrData) == 0:
        return "No QR Code Detected"
    else:
        isSecureQR = (isSecureQr(qrData[0]))
        # do something with isSecureQR 
    # context 
    import xml.etree.ElementTree as ET

# Parse the XML document
    for i in qrData:
        root = ET.fromstring(i)

        # Extract the data
        uid = root.attrib['uid']
        name = root.attrib['name']
        gender = root.attrib['gender']
        yob = root.attrib['yob']
        co = root.attrib['co']
        loc = root.attrib['loc']
        vtc = root.attrib['vtc']
        dist = root.attrib['dist']
        state = root.attrib['state']
        pc = root.attrib['pc']
    context = {"UID": uid,"Name": name,"Gender": gender,"YOB": yob}
    # Print the extracted data
        # print("UID:", uid)
        # print("Name:", name)
        # print("Gender:", gender)
        # print("YOB:", yob)
        # print("Co:", co)
        # print("Loc:", loc)
        # print("VTC:", vtc)
        # print("Dist:", dist)
        # print("State:", state)
        # print("PC:", pc)
        # return "File uploaded successfull"
    return render_template('index.html',context=context)


if __name__ == '__main__':
    app.run(debug=True)
