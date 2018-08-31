import Tkinter as tk
import util
import HaarFeature
import test
import Boosting
import Integrated
import FileDialog as fd
import cv2

class gv:
    face = []
    classifiers = []

window = tk.Tk()
window.title('Face Detection')
window.geometry('600x400')

def choosingface():
    file = fd.LoadFileDialog(window)
    filename = file.go('/Users/xavier0121/Desktop/camera/')
    image = cv2.imread(filename)
    gv.face = image

def training():
    gv.classifiers = util.train()

def detectface():
    test.detectfaces(gv.face, gv.classifiers)

b1 = tk.Button( window, text = 'Training DataSet', width = 15, height = 5, command = training )
b1.place(x = 100, y = 100)

b2 = tk.Button( window, text = 'Detection', width = 15, height = 5, command = detectface )
b2.place( x = 300, y = 200 )

b3 = tk.Button( window, text = 'Choose a face', width = 15, height = 5, command = choosingface )
b3.place( x = 300, y = 100 )

window.mainloop()