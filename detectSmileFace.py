import cv2
import cv2.cv as cv
import config
import picamera

def detect(image, cascade):
    haar_faces = cv2.CascadeClassifier(cascade)
    detected = haar_faces.detectMultiScale(image, scaleFactor=config.HAAR_SCALE_FACTOR, 
                minNeighbors=config.HAAR_MIN_NEIGHBORS, 
                minSize=config.HAAR_MIN_SIZE, 
                flags=cv.CV_HAAR_SCALE_IMAGE)
    if len(detected) == 0:
        return []
    detected[:,2:] += detected[:,:2]
    return detected


if __name__ == "__main__":
    with picamera.PiCamera() as camera:
            camera.capture('myface.jpg')
    
    image = cv2.imread('myface.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = detect(gray, config.HAAR_FACES)
 

    if len(faces) != 0:
        vis = image.copy()
        for (x,y,w,h) in faces:
            cv2.rectangle(vis, (x, y), (x+w, y+h),(0,255,0))
        for x1, y1, x2, y2 in faces:
            roi = gray[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]
            mouth = detect(roi.copy(), config.HAAR_MOUTH)

            for (x,y,w,h) in smiles:
                cv2.rectangle(vis_roi, (x, y), (x+w, y+h), (255,0,0))

        cv2.imwrite('detect.jpg', vis_roi)
    else:
        print 'No Face Detected'
