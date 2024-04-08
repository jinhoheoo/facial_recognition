# import cv2
# import dlib
# import numpy as np
# from keras.models import load_model

# prototxt_path = './models/MobileNetSSD_deploy.prototxt'
# model_path = './models/MobileNetSSD_deploy.caffemodel'
# net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", 
#            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#            "sofa", "train", "tvmonitor"]

# LABEL_COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

# expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# model = load_model('./models/emotion_model.hdf5')

# video_capture = cv2.VideoCapture('123.jpg')

# while True:
#     ret, frame = video_capture.read()
    
#     if not ret:
#         break

#     (h, w) = frame.shape[:2]
#     resized = cv2.resize(frame, (300, 300))
#     blob = cv2.dnn.blobFromImage(resized, 0.007843, (300, 300), 127.5)
    
#     net.setInput(blob)
#     detections = net.forward()

#     for i in np.arange(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
    
#         if confidence > 0.2:
#             idx = int(detections[0, 0, i, 1])
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
            
#             cv2.rectangle(frame, (startX, startY), (endX, endY), LABEL_COLORS[idx], 1)

#             # 얼굴 영역 추출
#             face_roi = frame[startY:endY, startX:endX]
#             gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

#             # 얼굴 영역에서 표정 인식
#             face_resized = cv2.resize(gray_face, (64, 64))
#             face_resized = np.expand_dims(face_resized, axis=-1)
#             face_resized = np.expand_dims(face_resized, axis=0)
#             face_resized = face_resized / 255.0

#             output = model.predict(face_resized)[0]
#             expression_index = np.argmax(output)
#             expression_label = expression_labels[expression_index]

#             # 표정 텍스트 표시
#             cv2.putText(frame, expression_label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # 프레임 출력
#     cv2.imshow('Expression Recognition', frame)

#     # ESC 키를 누르면 종료
#     key = cv2.waitKey(0)
#     if key == 27:
#         break

# if video_capture.Opened():
#     video_capture.release()
# cv2.destroyAllWindows()



import numpy as np
import cv2
import matplotlib.pyplot as plt

def img_show(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
        
        
prototxt_path = './models/MobileNetSSD_deploy.prototxt'
model_path = './models/MobileNetSSD_deploy.caffemodel'
 
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", 
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
 
LABEL_COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
img = cv2.imread("123.jpg")

(h, w) = img.shape[:2]
resized = cv2.resize(img, (300, 300))
blob = cv2.dnn.blobFromImage(resized, 0.007843, (300, 300), 127.5)
 
net.setInput(blob)
detections = net.forward()

vis = img.copy()
conf = 0.2

for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
 
    if confidence > conf:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        print("[INFO] {} : [ {:.2f} % ]".format(CLASSES[idx], confidence * 100))
        
        cv2.rectangle(vis, (startX, startY), (endX, endY), LABEL_COLORS[idx], 1)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(vis, "{} : {:.2f}%".format(CLASSES[idx], confidence * 100), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LABEL_COLORS[idx], 2)
        

img_show('Object Detection', vis, figsize=(16,10))