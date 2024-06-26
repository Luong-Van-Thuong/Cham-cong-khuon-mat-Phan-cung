import os
import sqlite3
from datetime import datetime
import pandas as pd
from collections import Counter
# Đường dẫn tới cơ sở dữ liệu
current_directory = os.getcwd()
file_sql2 = "LuuThoiGian.db"
datafile2 = os.path.join(current_directory, file_sql2)

# Kết nối tới cơ sở dữ liệu
ketNoiData2 = sqlite3.connect(datafile2)
cursor1 = ketNoiData2.cursor()

# Tạo bảng nếu chưa tồn tại
cursor1.execute('''
    CREATE TABLE IF NOT EXISTS test(
        id INTEGER,
        name TEXT, 
        thoigianvao TEXT,
        thoigianvao2 TEXT,
        thoigianra TEXT,
        thoigianra2 TEXT,
        ngaythangnam INTEGER
    );
''')

# Hàm để kiểm tra và cập nhật thời gian
def update_time_for_id(id, name):
    # Lấy thời gian hiện tại
    now = datetime.now()
    current_date = int(now.strftime('%Y%m%d'))
    current_time = now.strftime('%H:%M')

    # Kiểm tra xem có hàng nào có ngaythangnam và id phù hợp không
    cursor1.execute("SELECT * FROM test WHERE ngaythangnam=? AND id=?", (current_date, id))
    row = cursor1.fetchone()

    if row is None:
        # Tạo mới hàng nếu chưa tồn tại
        cursor1.execute('''
            INSERT INTO test (id, name, ngaythangnam, thoigianvao, thoigianvao2, thoigianra, thoigianra2)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (id, name, current_date, None, None, None, None))
        ketNoiData2.commit()
        # Lấy lại hàng mới tạo
        cursor1.execute("SELECT * FROM test WHERE ngaythangnam=? AND id=?", (current_date, id))
        row = cursor1.fetchone()

    # Cập nhật thời gian dựa trên giờ hiện tại
    if '06:45' <= current_time <= '07:30':
        if row[2] is None:  # thoigianvao
            cursor1.execute("UPDATE test SET thoigianvao=? WHERE id=? AND ngaythangnam=?", (current_time, id, current_date))
    elif '07:31' <= current_time <= '17:00':
        if row[3] is None:  # thoigianvao2
            cursor1.execute("UPDATE test SET thoigianvao2=? WHERE id=? AND ngaythangnam=?", (current_time, id, current_date))
    elif '17:01' <= current_time <= '17:30':
        if row[4] is None:  # thoigianra
            cursor1.execute("UPDATE test SET thoigianra=? WHERE id=? AND ngaythangnam=?", (current_time, id, current_date))
    elif '17:31' <= current_time <= '23:00':
        if row[5] is None:  # thoigianra2
            cursor1.execute("UPDATE test SET thoigianra2=? WHERE id=? AND ngaythangnam=?", (current_time, id, current_date))

    ketNoiData2.commit()
    # # Lấy thời gian hiện tại
    # now = datetime.now()
    # current_date = now.strftime('%Y%m%d')
    # current_time = now.time()

    # # Cấu hình kết nối để trả về các hàng dưới dạng dictionary
    # cursor1.row_factory = sqlite3.Row

    # # Kiểm tra nếu id đã tồn tại
    # cursor1.execute("SELECT * FROM test WHERE id = ?", (id,))
    # record = cursor1.fetchone()

    # if not record:
    #     # Nếu id chưa tồn tại, tạo một hàng mới
    #     cursor1.execute("INSERT INTO test (id, name, ngaythangnam) VALUES (?, ?, ?)", (id, name, current_date))
    #     ketNoiData2.commit()  # Lưu thay đổi để đảm bảo hàng mới được tạo
    # else:
    #     # Lấy giá trị hiện tại của các cột
    #     thoigianvao = record['thoigianvao']
    #     thoigianvao2 = record['thoigianvao2']
    #     thoigianra = record['thoigianra']
    #     thoigianra2 = record['thoigianra2']

    # # Kiểm tra khoảng thời gian và cập nhật cột thích hợp nếu cột đó chưa có giá trị
    # if current_time >= datetime.strptime('07:00', '%H:%M').time() and current_time <= datetime.strptime('07:15', '%H:%M').time():
    #     if not thoigianvao:
    #         cursor1.execute("UPDATE test SET thoigianvao = ? WHERE id = ?", (now.strftime('%H:%M:%S'), id))
    # elif current_time >= datetime.strptime('08:00', '%H:%M').time() and current_time <= datetime.strptime('17:00', '%H:%M').time():
    #     if not thoigianvao2:
    #         cursor1.execute("UPDATE test SET thoigianvao2 = ? WHERE id = ?", (now.strftime('%H:%M:%S'), id))
    # elif current_time >= datetime.strptime('17:00', '%H:%M').time() and current_time <= datetime.strptime('17:15', '%H:%M').time():
    #     if not thoigianra:
    #         cursor1.execute("UPDATE test SET thoigianra = ? WHERE id = ?", (now.strftime('%H:%M:%S'), id))
    # elif current_time >= datetime.strptime('17:15', '%H:%M').time() and current_time <= datetime.strptime('22:00', '%H:%M').time():
    #     if not thoigianra2:
    #         cursor1.execute("UPDATE test SET thoigianra2 = ? WHERE id = ?", (now.strftime('%H:%M:%S'), id))
    # else:
    #     print("Thời gian hiện tại không nằm trong khoảng thời gian đã chỉ định.")

    # # Lưu thay đổi
    # ketNoiData2.commit()

# update_time_for_id(1, 'Nguyen Van A')
 
# # Đóng kết nối
# ketNoiData2.close()
def export_data_to_excel():
    """
    Đọc dữ liệu từ cơ sở dữ liệu SQLite và lưu vào file Excel.

    Parameters:
    - database_path: Đường dẫn đến file cơ sở dữ liệu SQLite.
    - table_name: Tên bảng trong cơ sở dữ liệu cần đọc dữ liệu.
    - output_file: Đường dẫn và tên file Excel để lưu dữ liệu.
    """
    database_path = datafile2
    table_name = 'test'
    output_file = 'bang_cham_cong.xlsx'    
    try:
        # Kết nối đến cơ sở dữ liệu SQLite
        conn = sqlite3.connect(database_path)

        # Đọc dữ liệu từ bảng
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)

        # Lưu dữ liệu vào file Excel
        df.to_excel(output_file, index=False)

        # Đóng kết nối cơ sở dữ liệu
        conn.close()

        print(f'Dữ liệu đã được lưu vào file {output_file}')
    except Exception as e:
        print(f'Đã xảy ra lỗi: {e}')

# Ví dụ sử dụng hàm

# export_data_to_excel()


# Kiem tra so lan xuat hien
def kiemtra_so_lan_xuat_hien(ds):
    tan_suat = Counter(ds)
    chuoi_pho_bien, _ = tan_suat.most_common(1)[0]  # Lấy chuỗi phổ biến và bỏ qua tần suất
    name = chuoi_pho_bien.split('_')[0]
    id = chuoi_pho_bien.split('_')[1] 
    update_time_for_id(id, name) 
    export_data_to_excel()
    return id, name