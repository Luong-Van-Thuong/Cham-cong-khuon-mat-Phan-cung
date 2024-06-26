import os
from os import listdir
from PIL import Image as Img
from numpy import asarray
from numpy import expand_dims
# from matplotlib import pyplot
# from keras.models import load_model
from keras_facenet import FaceNet
import pickle
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import sql_nhan_vien as ttnv

HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()

folder='Images/'
current_directory = os.getcwd()
global database 
database = {}
global soluonganh


def train_tb_ttnv():
    tttbnv = ttnv.lay_tt_nv()
    for row in tttbnv:
        user_name = row[1].replace(" ", "_")
        user_id = row[0]
        # ---------------Tao ten name_folder_image---------------------------
        name_folder_image = f"{user_name}_{user_id}"
        name_pkl = f"{row[1]}_{row[0]}"
        # ---------------Tao duong dan thu muc name_folder_image---------------------------
        path = folder + name_folder_image
        database1 = []
        for filename in list(os.listdir(path)):
            image_path = os.path.join(path, filename)  # Tạo đường dẫn đầy đủ đến tệp ảnh
            gbr1 = cv2.imread(image_path)  # Đọc tệp ảnh bằng đường dẫn đầy đủ
            wajah = HaarCascade.detectMultiScale(gbr1,1.1,4)
            if len(wajah)>0:
                x1, y1, width, height = wajah[0]         
            else:
                x1, y1, width, height = 1, 1, 10, 10
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height            
            gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
            gbr = Image.fromarray(gbr)                  # konversi dari OpenCV ke PIL
            gbr_array = asarray(gbr)
            face = gbr_array[y1:y2, x1:x2]                        
            face = Image.fromarray(face)                       
            face = face.resize((160,160))
            face = asarray(face)
            face = expand_dims(face, axis=0)
            signature = MyFaceNet.embeddings(face)
            database1.append(signature)

        average_embedding = np.mean(database1, axis=0)    
        database[os.path.splitext(name_pkl)[0]]=average_embedding
        
    print(database)
    myfile = open("data.pkl", "wb")
    pickle.dump(database, myfile)
    myfile.close()


def phan_biet_nv():
    myfile = open("data.pkl", "rb")
    database = pickle.load(myfile)
    myfile.close()
    cap = cv2.VideoCapture(0)

    while(1):
        _, gbr1 = cap.read()
        #----------------------Tìm khuôn mặt trong frame----------------------------
        wajah = HaarCascade.detectMultiScale(gbr1,1.1,4)
        
        if len(wajah)>0:
            x1, y1, width, height = wajah[0]        
        else:
            x1, y1, width, height = 1, 1, 10, 10
        
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        #-------------------Chuyển đổi màu--------------------------
        gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
        gbr = Image.fromarray(gbr)                  # konversi dari OpenCV ke PIL
        gbr_array = asarray(gbr)
        
        face = gbr_array[y1:y2, x1:x2]                        
        #-----------------Chuyển đổi thảnh mảng---------------------
        face = Image.fromarray(face)                       
        face = face.resize((160,160))
        face = asarray(face)
        
        # face = face.astype('float32')
        # mean, std = face.mean(), face.std()
        # face = (face - mean) / std
        # -------------------Thêm chiều để đủ 4 chiều trước khi đưa vào FaceNet----------------
        face = expand_dims(face, axis=0)
        signature = MyFaceNet.embeddings(face)
        
        min_dist=100
        identity=' '
        for key, value in database.items() :
            tt_key = key.split('_')
            tt_name = tt_key[0]
            tt_id = tt_key[1]
            dist = np.linalg.norm(value-signature)
            # print("Gia tri tong khoang cach: ",dist)
            if dist < min_dist:
                min_dist = dist
                identity = key
                
        print("Ten du doan: ", identity, " Gia tri dist: ", dist)
        cv2.putText(gbr1,identity, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(gbr1,(x1,y1),(x2,y2), (0,255,0), 2)
            
        cv2.imshow('res',gbr1)
        
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
            
    cv2.destroyAllWindows()
    cap.release()

#---------------------------------Chạy chương trình------------------------------------------

# train_tb_ttnv()      
phan_biet_nv()  