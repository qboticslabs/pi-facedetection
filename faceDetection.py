import cv2
import cv2.cv as cv
import config
import picamera

def detect_faces(image):
    haar_faces = cv2.CascadeClassifier(config.HAAR_FACES)
    detected = haar_faces.detectMultiScale(image, scaleFactor=config.HAAR_SCALE_FACTOR, 
                minNeighbors=config.HAAR_MIN_NEIGHBORS, 
                minSize=config.HAAR_MIN_SIZE, 
                flags=cv2.CASCADE_SCALE_IMAGE)
    
    return detected

if __name__ == "__main__":
    with picamera.PiCamera() as camera:
            camera.capture('myface.jpg')
    
    image = cv2.imread('myface.jpg')

    faces = detect_faces(image)
    if len(faces) != 0:
        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), 255)

        cv2.imwrite('detect.jpg', image)
    else:
        print 'No Face Detected'
