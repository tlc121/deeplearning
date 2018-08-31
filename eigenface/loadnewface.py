#coding=utf-8
import cv2

def captureimage(name):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('test')
    i = 0
    while True:
        key = cv2.waitKey(1)
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow('test', frame)
        # press esc to quit
        if key % 256 == 27:
            print 'esc'
            break
        # press space to save image
        elif key % 256 == 32:
            cv2.imwrite('/Users/xavier0121/Desktop/camera/s2' + str(i) + '.pgm', frame)
            print str(i)
            i += 1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    captureimage('gary')