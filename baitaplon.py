import cv2
import face_recognition
import os #dùng để load toàn bộ kho ảnh
import numpy as np
# b1 load anh tu kho anh pic 2
path="pic2"
images = []
className = []
mylist =os.listdir(path)# kiểm tra toàn bộ tên file ảnh trong thư viện pic2
print(mylist)
for cl in mylist: # lấy ra tên của file ảnh
    curimg = cv2.imread(f"{path}/{cl}")# đọc từng bức ảnh từ cv2 và đẩy về ma trận
    images.append(curimg)#thêm toàn bộ ma trận điểm ảnh vào images
    className.append(os.path.splitext(cl)[0])#tách tưng đoạn tên của tưng file trong danh sách ảnh đang chạy theo dấu chấm
print(len(images))
print(images)
print(className)
# b2 ma hoa cac anh
def Mahoa(images):# mã hóa list ma trânj
    encodeList = []#tạo hàm rỗng vì khi load list ảnt thì phải đẩy từng ảnh vaod
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# chuyển ảnh bgr sang dạng rgb
        # vì thư viện opcv mặc đinh là dang bgr hiện nay thì ảnh là ở dang rgb nên chuyển để tránh sai lệch màu sắc
        encode = face_recognition.face_encodings(img)[0]# mã hóa từng bức ảnh lấy giá trị 0 vì là load từng bức ảnh nên vi trí mã hóa là 0
        encodeList.append(encode)#đẩy các giá trị mã hóa vào encodelist
    return encodeList
encodeListKnow = Mahoa(images) #gán các giá trị vừa mã hóa vào encode...
print("ma hoa thanh cong")
print(len(encodeListKnow))
# khoi dong webcam
cap= cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()# cap.read trả về 2 giá trị là ret và frame và ret là trả về giá trị true khi came load đc ảnh và false khi cam không load đc
    frameS = cv2.resize(frame,(0,0),None,fx=0.5,fy=0.5) # thay đổi khung ảnh ban đầu với
    frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)# chuyển về dang rgb

    # xac dinh vi tri khuan mat
    facecurFrame = face_recognition.face_locations(frameS) # lay tung frame ảnh trong video vao vi tri khuan mat hien tai
    encodecurFrame = face_recognition.face_encodings(frameS)#mã hóa từng frame ảnh tại thời điển hiện tại
    for encodeFace, faceLoc in zip(encodecurFrame,facecurFrame):#chạy 2 biến encde và face để chạy ghép cặp giữa vị trí ảnh bất kì và đoạn mã hóa tại thời điểm đó
        #matsches = face_recognition.compare_faces(encodeListKnow,encodeFace)#so sánh 2 cái mã hóa trong kho ảnh mà mã hóa ảnh tại cam
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)#so sánh sự khác nhau giữa 2 đaonj mã hóa
        print(faceDis)
        matchIndex = np.argmin(faceDis) # day ve gia tri index nho nhat

        if faceDis[matchIndex] <0.5 :
            name = className[matchIndex].upper()
        else:
            name = " nguoi la"
        # ve ten len anh
        y1, x2,y2,x1= faceLoc
        y1, x2, y2, x1= y1*2, x2*2,y2*2,x1*2 # vẽ ô vuông nhận dạng mặt vì tỉ lêk ảnh lấy 0.5 nên ô vuông phải nhân lên 2
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,2,255),2)
        cv2.putText(frame,name,(x2,y2),cv2.FONT_HERSHEY_COMPLEX, 1,(155,255,255),2)
    cv2.imshow('cam quan sat', frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()