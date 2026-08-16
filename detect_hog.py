import cv2
import time
import face_recognition

# Mo webcam cua laptop
cap = cv2.VideoCapture(0)

prev_time = 0

print("Dang mo Webcam voi thuat toan HOG... Nhan 'q' de thoat.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Khong tim thay Webcam!")
        break

    # Thu nho kich thuoc khung hinh de tang toc do xu ly (Rat quan trong khi chay tren Pi sau nay)
    # O day minh thu nho xuong 1/4
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # QUAN TRONG: OpenCV su dung he mau BGR, nhung thu vien face_recognition yeu cau he mau RGB
    # Chung ta phai chuyen doi he mau thi no moi nhan dien chinh xac
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Dua anh RGB vao thuat toan HOG de tim vi tri khuon mat
    # Ham nay tra ve mot danh sach cac toa do (top, right, bottom, left) cua cac khuon mat
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

    # Ve khung chu nhat quanh cac khuon mat tim duoc
    for (top, right, bottom, left) in face_locations:
        # Vi luc nay chung ta thu nho anh 1/4 de xu ly, gio phai nhan 4 toa do len de ve dung tren anh goc
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, "Khuon mat (HOG)", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Tinh toan va hien thi FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Hien thi video
    cv2.imshow('He thong Detect - HOG (Dlib)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()