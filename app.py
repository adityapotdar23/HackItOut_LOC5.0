from flask import Flask, render_template, request
import cv2
from pyaadhaar.utils import Qr_img_to_text, isSecureQr
import cv2
from pyaadhaar.utils import Qr_img_to_text, isSecureQr
from pyaadhaar.decode import AadhaarSecureQr
from pyaadhaar.decode import AadhaarOldQr
import xml.etree.ElementTree as ET
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


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

        cv2.imwrite('image.png', img[y-10:y+h+10, x-10:x+w+10])
        # Extract Aadhaar card details from the image
        qrData = Qr_img_to_text('image.png')
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

    return render_template('aadhar.html', context=context)


if __name__ == '__main__':
    app.run(debug=True)
