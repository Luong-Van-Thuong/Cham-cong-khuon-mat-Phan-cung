import numpy as np
import os
import pickle, sqlite3
from PIL import Image, ImageTk
from datetime import datetime
import cv2
#---------------------------------------KHAI BAO BIEN---------------------------------------------
######################################################################################################################################################
#--------------------------------CÁC LỆNH KHỞI TẠO DATABASE, FOLDER CƠ BẢN----------------------------------------
# Khoi tạo đường dẫn đến file
current_directory = os.getcwd()

# Khởi tạo cơ sở dữ liệu người dùng
file_sql = "DuLieuNguoiDung1.db"
datafile = os.path.join(current_directory, file_sql)
ketNoiData = sqlite3.connect(datafile)
cursor = ketNoiData.cursor()
cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS Person(
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT
    );
''')
def them_ten_nv():
    # Nhập tên nhân viên vào bảng Person
    ten_nhan_vien = input("Nhập tên nhân viên: ")
    # Thực hiện lệnh INSERT để thêm tên nhân viên vào bảng
    cursor.execute("INSERT INTO Person (Name) VALUES (?)", (ten_nhan_vien,))
    # Lưu các thay đổi và đóng kết nối đến cơ sở dữ liệu
    ketNoiData.commit()
    ketNoiData.close()
    print("Đã thêm nhân viên thành công vào bảng Person.")

def lay_tt_nv():
    # Lấy toàn bộ dữ liệu từ bảng Person
    cursor.execute("SELECT ID, Name FROM Person")
    # Lấy tất cả các dòng dữ liệu từ kết quả truy vấn
    rows = cursor.fetchall()
    return rows
    # # In ra từng cặp ID và Name
    # for row in rows:
    #     print(f"ID: {row[0]}, Name: {row[1]}")

    # Đóng kết nối đến cơ sở dữ liệu

# -------------------------Chạy chuong trinh--------------------------------------
them_ten_nv()
# rows = lay_tt_nv()
# for row in rows:
#     print(f"ID: {row[0]}, Name: {row[1]}")
