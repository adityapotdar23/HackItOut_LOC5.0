from flask import Flask, render_template, request
import cv2
from pyaadhaar.utils import Qr_img_to_text, isSecureQr
from pyaadhaar.decode import AadhaarSecureQr
from pyaadhaar.decode import AadhaarOldQr
import xml.etree.ElementTree as ET
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from cryptography.fernet import Fernet
import qrcode
import re
import easyocr
from pyzbar.pyzbar import decode
import uuid
from flask import Flask, send_file
from io import BytesIO

yob = None 
name = None

app = Flask(__name__)

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])

pan_number = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign')
def sign():
    return render_template('sign.html')

name = None 
yob = None 
@app.route("/aadhar", methods=['GET', 'POST'])
def aadhar():
    global name 
    global yob 
    if request.method == 'POST':
        # Get the uploaded file from the HTML form
        file = request.files['aadhar_img']
        # Read the image file using OpenCV 

        status = True
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
            context = {"aadhar_status":"Upload another Aadhar QR Code image"}

        else:
            # Parse the XML document
            for i in qrData:
                root = ET.fromstring(i)
                uid = root.attrib['uid']
                name = root.attrib['name'].lower()
                gender = root.attrib['gender']
                yob = root.attrib['yob'] 
                context = {"UID": uid, "Name": name,
                           "Gender": gender, "YOB": yob, "aadhar_status":"Aadhar code detected"} 
    else: 
        img_name = None 
        context = {"aadhar_status":"Upload another Aadhar QR Code image"}

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
            print(mylst_1)
            print(name,yob)
            status = 'Reupload your correct PAN CARD image' 
            if 'income' in mylst_1[0]:
                status = " PAN CARD image uploaded successully" 
            if name in mylst_1[0] and yob in mylst_1[0]:
                verified = 'PAN CARD verified'
            else:
                verified = 'Re-upload PAN CARD. Not Verified' 
            context = {'pan_card_status': status, 'verified_status':verified}

    else:
        context = None
        pan_img = None

    return render_template('pan.html', context=context, pan_img=pan_img)

@app.route('/digitalid', methods=['GET', 'POST'])
def digitalid():
    global name
    global yob
    if request.method == 'POST':
        def uuid_maker(data1):
            file = request.files['image']
            # generate a UUID and truncate it to the first 8 characters
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            cv2.imwrite('static/images/digitalpic.jpg', img)
            '''
            data1: list of data
            '''
            users_list = data1
            usrconfig = ET.Element("usrconfig")
            # create sub element
            usrconfig = ET.SubElement(usrconfig, "usrconfig")
            # insert list element into sub elements
            for user in range(len(users_list)):
                usr = ET.SubElement(usrconfig, "usr")
                usr.text = str(users_list[user])
            tree = ET.ElementTree(usrconfig)
            # write the tree into an XML file
            tree.write("static/xml/Output.xml", encoding ='utf-8', xml_declaration = True)
            with open('static/xml/Output.xml', 'r') as f:
                message = f.read()
            font1 = ImageFont.truetype("static/OpenSans-Semibold.ttf", size=45)
            font2 = ImageFont.truetype("static/OpenSans-Semibold.ttf", size=55)
                        
            # made the QRcode for encoded data
            # define the string to encode as a QR code
        #     string = r'''<?xml version='1.0' encoding='utf-8'?><usrconfig><usr>Sarvagya Singh</usr><usr>18/12/2002</usr><usr>Mumbai Maharashtra</usr><usr>photo1.jpg</usr></usrconfig>'''

            # create a QR code instance
            qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
            # add the data to the QR code instance
            qr.add_data(message)
            # make the QR code
            qr.make(fit=True)
            # create an image from the QR code instance
            img = qr.make_image(fill_color="black", back_color="white")
            # display the image
            img.save("static/img/EncodedQR.png")
            
            
            # added data on template     
            template = Image.open("static/img/uuid_template.png")
            pic = Image.open(r"static/images/digitalpic.jpg").resize((265, 360), Image.ANTIALIAS)
            template.paste(pic, (35, 90, 300, 450))
            draw = ImageDraw.Draw(template)
            # yob
            draw.text((540, 310), str(data1[3]), font=font1, fill='black')
            draw.text((480, 200), data1[0], font=font2, fill='black')
            pic = Image.open(f"static/img/EncodedQR.png").resize((200, 200), Image.ANTIALIAS)

            template.paste(pic, (550, 390,750,590))
            draw = ImageDraw.Draw(template)
            
            #     Saving the file
            templated = cv2.cvtColor(np.array(template), cv2.COLOR_BGR2RGB)
            # decoded = decode(template)
            # print(decoded[0].data.decode())
            return template
        
        my_uuid = str(uuid.uuid4().hex)[:10]
        my_uuid = my_uuid.upper()
        data = [name[:14].upper(),yob,"static/img/photo1.jpg",my_uuid]
        digitalid_img = uuid_maker(data)
        # np_img = 
        templated = cv2.cvtColor(np.array(digitalid_img),  cv2.COLOR_BGR2RGB)
        cv2.imwrite('static/images/digitalid.png',templated)
        digitalid_img = 'digitalid.png'
        # cv2.imwrite('static/images/digitalid.png',digitalid_img)
    else:
        digitalid_img = None
    return render_template('digitalid.html', digitalid_img=digitalid_img)



@app.route('/voter', methods=['GET', 'POST'])
def voter():
    global name
    global yob
    if request.method == 'POST':
        # Get the uploaded file from the HTML form
        file = request.files['voter_img']
        # Read the image file using OpenCV
        img = cv2.imdecode(np.fromstring(
            file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        cv2.imwrite('static/images/voter_img.png', img)

        voter_img = 'voter_img.png'
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
            status = 'Reupload your correct Voter ID image' 
            if 'election' in mylst_1[0]:
                status = "Voter ID image uploaded successully" 
            if name in mylst_1[0] and yob in mylst_1[0]:
                verified = 'Voter ID verified'
            else:
                verified = 'Re-upload Voter ID. Not Verified' 
            context = {'voter_id_status': status, 'verified_status':verified}

    else:
        context = None
        voter_img = None

    return render_template('voter.html', context=context, voter_img=voter_img)

@app.route('/convert-image-to-pdf')
def convert_image_to_pdf():
    # Open the image file
    img = Image.open('static/images/digitalid.png')
    print(img)
    # Convert the image to PDF format
    buffer = BytesIO()
    img.save(buffer, 'PDF')
    
    # Set the buffer's cursor to the beginning
    buffer.seek(0)
    
    # Return the PDF file as a downloadable attachment
    return send_file(buffer, download_name='didital_identity.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
