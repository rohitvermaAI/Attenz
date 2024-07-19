from flask import Flask, request, render_template, jsonify
import os
import cv2
import base64
import datetime
import csv
import numpy as np
from PIL import Image
import face_recognition
from io import BytesIO

import time
import pickle
import pandas as pd
from geopy.distance import geodesic

app = Flask(__name__)
from subprocess import STDOUT, check_call
check_call(['apt-get', 'update'], stdout=open(os.devnull,'wb'), stderr=STDOUT)
check_call(['apt-get', 'install', '-y', 'libgl1'], stdout=open(os.devnull,'wb'), stderr=STDOUT)
check_call(['apt-get', 'install', '-y', 'libglib2.0-0'], stdout=open(os.devnull,'wb'), stderr=STDOUT)
check_call(['apt-get', 'update'], stdout=open(os.devnull,'wb'), stderr=STDOUT)

def save_image(image_data, eid, ename):
    img_path = f"static/TrainingImage/{ename}.{eid}.png"
    image_data = image_data.split(",")[1]  # Remove the header of the base64 string
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = img.resize((160, 160), Image.LANCZOS)
    img.save(img_path)
    save_details(eid, ename)
    return "Image captured and saved", 200

def save_details(eid, ename):
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    time_str = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    csv_path = 'EmployeeDetails/EmployeeDetails.csv'

    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['ID', 'Name', 'Date', 'Time'])
    
    with open(csv_path, 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([eid, ename, date, time_str])
def getImagesAndLabels(path):
    emp_images = os.listdir(path)
    faceSamples = []
    Ids = []
    for image in emp_images:
        Id = int(os.path.split(image)[-1].split(".")[1])
        faceSamples.append(cv2.imread(os.path.join(path, image)))
        Ids.append(Id)
    return faceSamples, Ids

def trainimg():
    try:
        faces, Ids = getImagesAndLabels("static/TrainingImage")
    except Exception as e:
        return str(e), 500
    
    encodeList = []
    for face in faces:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(face)[0]
        encodeList.append(encode)

    encodeListKnownWithIds = [encodeList, Ids]
    with open("EncodeFile.p", 'wb') as file:
        pickle.dump(encodeListKnownWithIds, file)

    return "Model trained and saved", 200

def Fillattendance(image_data):
    try:
        with open('EncodeFile.p', 'rb') as file:
            encodeListKnownWithIds = pickle.load(file)
        encodeListKnown, Ids = encodeListKnownWithIds
    except FileNotFoundError:
        return "Encode file not found", 404
    
    img = Image.open(BytesIO(base64.b64decode(image_data.split(",")[1])))
    img = img.resize((160, 160), Image.LANCZOS)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    today_date = datetime.datetime.now().strftime('%d-%m-%Y')
    csv_path = f'Attendance/attendance_{today_date}.csv'
    faceCurFrame = face_recognition.face_locations(img)
    encodeCurFrame = face_recognition.face_encodings(img, faceCurFrame)
    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                id = Ids[matchIndex]
                df = pd.read_csv("EmployeeDetails/EmployeeDetails.csv")
                aa = df.loc[df['ID'] == id]['Name'].values[0]
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                today_date = datetime.datetime.now().strftime('%d-%m-%Y')
                csv_path = f'Attendance/attendance_{today_date}.csv'
                fieldnames = ['ID', 'Name', 'In_Time', 'Out_Time', 'Total Hours']

                if not os.path.exists(csv_path):
                    with open(csv_path, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                with open(csv_path, 'a+', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    attendance_data = {'ID': id, 'Name': aa, 'In_Time': current_time, 'Out_Time': '', 'Total Hours': ''}
                    writer.writerow(attendance_data)
                image_path = f"static/TrainingImage/{aa}.{id}.png"
                return jsonify({'status': 'success', 'id': id, 'name': aa, 'image_path': image_path}), 200

    return jsonify({'status': 'error', 'message': 'No face found'}), 400

    return "No face found", 400

def check_loc():
    office_coords = (40.7128, -74.0060)
    user_coords = (40.7128, -74.0060)
    distance = geodesic(office_coords, user_coords).meters
    return distance <= 20

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_employee')
def new_employee():
    return render_template('capture.html')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    eid = request.form.get('eid')
    ename = request.form.get('ename')
    csv_path = 'EmployeeDetails/EmployeeDetails.csv'

    if os.path.exists(csv_path):
        with open(csv_path, 'r') as csvFile:
            reader = csv.DictReader(csvFile)
            for row in reader:
                if row['ID'] == eid:
                    return '''
                        <script>
                            alert("Employee ID already exists");
                            window.location.href = "/new_employee";
                        </script>
                    '''
    image_data = request.form.get('image_data')
    if not eid or not ename or not image_data:
        return  '''
                        <script>
                            alert("Employee ID and Employee Name required");
                            window.location.href = "/new_employee";
                        </script>
                    '''
    return save_image(image_data, eid, ename)

@app.route('/train_model')
def train_model():
    return render_template('train.html')

@app.route('/train_img', methods=['GET'])
def train_model_post():
    return trainimg()

@app.route('/take_attendance')
def take_attendance():
    return render_template('attendance.html')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance_post():
    image_data = request.form.get('image_data')

    return Fillattendance(image_data)

@app.route('/check_location')
def check_location():
    return render_template('location.html')

@app.route('/check_location', methods=['POST'])
def check_location_post():
    if check_loc():
        return "You are within the office vicinity", 200
    else:
        return "You are not around the office", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
