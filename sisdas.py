import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from pygame import mixer
import os

mixer.init()
sound = mixer.Sound('alarm.wav')

# Menginisialisasi face detector dan facial landmark predictor 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
thicc=2
path = os.getcwd()
cap=cv2.VideoCapture(0)
font=cv2.FONT_HERSHEY_SIMPLEX

# mendefinisikan titik tengah
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

# mendifinisikan jarak Euclidean
def euclidean_distance(leftx,lefty, rightx, righty):
  return np.sqrt((leftx-rightx)**2 +(lefty-righty)**2)

# menentukan aspek resio mata
def get_EAR(eye_points, facial_landmarks):
    # mendifinisikan titik mata kiri
    left_point = [facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y]
    # mendifinisikan titik mata kanan   
    right_point = [facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y]
    # mendifinisikan titik tengah mata atas    
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    # mendifinisikan titik tengah mata bawah   
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    # menggambar garis horizontal dan vertikal      
    hor_line = cv2.line(frame, (left_point[0], left_point[1]), (right_point[0], right_point[1]), (255, 0, 0), 3)
    ver_line = cv2.line(frame, (center_top[0], center_top[1]),(center_bottom[0], center_bottom[1]), (255, 0, 0), 3)
    # menghitung panjang garis horizontal dan vertikal   
    hor_line_lenght = euclidean_distance(left_point[0], left_point[1], right_point[0], right_point[1])
    ver_line_lenght = euclidean_distance(center_top[0], center_top[1], center_bottom[0], center_bottom[1])
    # menghitung aspek rasio mata     
    EAR = ver_line_lenght / hor_line_lenght
    return EAR

# variabel untuk sinyal mata tertutup
eye_close_signal=[]
# variabel jumlah score 
score = 0

# memulai looping
while True:
    ret, frame = cap.read() 
    height,width = frame.shape[:2]

    if ret == False:
        break
    # menkonversi warna fram ke grayscale  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # membuat objek untuk deteksi wajah  
    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        # membuat objek untuk mendeteksi landmark wajah
        landmarks = predictor(gray, face)
        # menghitung aspek rasio mata kiri    
        left_eye_ratio = get_EAR([36, 37, 38, 39, 40, 41], landmarks)
        # menhitung aspek rasio mata kanan  
        right_eye_ratio = get_EAR([42, 43, 44, 45, 46, 47], landmarks)
        # menghitung aspek rasio kedua mata  
        eye_close_ratio = (left_eye_ratio + right_eye_ratio) / 2
        # membulatkan rasio tertutupnya mata dari dua tempat(atas dan bawah) dengan decimal   
        eye_close_ratio_1 = eye_close_ratio * 100
        eye_close_ratio_2 = np.round(eye_close_ratio_1)
        eye_close_ratio_rounded = eye_close_ratio_2 / 100
        # menambah rasio tertutupnya mata ke variabel eye_close
        eye_close_signal.append(eye_close_ratio)
        if eye_close_ratio < 0.20:      
            score = score + 1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)
        else:
            score = score - 1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)
        #jika skornya kurang dari nol maka anggap nilainya nol (mata terbuka)
        if score <0:
            score = 0;        
        #jika skornya lebih besar dari 15 maka artinya mata pengemudi tertutup untuk waktu yang cukup lama
        if(score>15):
            #membunyikan alarm 
            cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            try:
                sound.play()
                
            except:  # jika suara = False
                pass
            #nilai untuk menambah offset frame kedalam (danger display)
            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)

        # menampilkan jumlah skor      
        cv2.putText(frame, str(score), (30, 50), font, 2, (0, 0, 255),5)
        cv2.putText(frame, str(eye_close_ratio_rounded), (900, 50), font, 2, (0, 0, 255),5)
        #menampilkan fram
        cv2.imshow('Face Detection', frame)
    
    #perintah untuk menghentikan program
    k = cv2.waitKey(20) & 0xff
    if k == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()