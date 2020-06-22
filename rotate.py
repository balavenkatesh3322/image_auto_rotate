import cv2
import numpy as np


def detect_face(frame):

    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0, (300,300), (104.0,177.0,123.0))

    net.setInput(blob)

    faces = net.forward()

    for i in range(0, faces.shape[2]):
        confidence = faces[0,0,i,2]
        
        if confidence < 0.7:
            continue

        box = faces[0,0,i,3:7] * np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype('int')

        text = "face " + "{:.2f}%".format(confidence * 100)        

        #cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        #cv2.putText(frame, text, (startX,startY-10), cv2.FONT_HERSHEY_SIMPLEX,
                    #1, (0,255,0), 2)
        cv2.imwrite('test.jpg',frame)
        return True

def rotate_image(frame,center,scale,angle):
    (h, w) = frame.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, scale)
    frame = cv2.warpAffine(frame, M, (h, w))
    return detect_face(frame)

def main():    
    frame = cv2.imread('6.jpg')
    original_status = detect_face(frame)
    print(original_status,'original_status')
    (h, w) = frame.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    scale = 1.0
    angle_90 = 90
    angle_180 = 180
    angle_270 = 270
    if original_status is None:
        status_90 = rotate_image(frame,center,scale,angle_90)
        print(status_90,'status_90')
        if status_90 is None:
            status_180 = rotate_image(frame,center,scale,angle_180)
            print(status_180,'status_180')
            if status_180 is None:
                status_270 = rotate_image(frame,center,scale, angle_270)
                print(status_270,'status_270')
    

if __name__ == "__main__":
    main()