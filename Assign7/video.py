import cv2
def main():
    sample_img = cv2.imread('/Users/xavier0121/Desktop/Video/s2.jpg')
    width, length, num = sample_img.shape

    img_root = '/Users/xavier0121/Desktop/Video/'
    fps = 24

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('/Users/xavier0121/Desktop/saveVideo.avi', fourcc, fps, (length, width))

    for i in range(2, 400):
        j = str(i)
        frame = cv2.imread(img_root + 's' + j + '.jpg')
        videoWriter.write(frame)

    videoWriter.release()

main()