import Tkinter as tk
import eigenface
import tkFileDialog as tfd
import FileDialog as fd
import cv2
import tkMessageBox as tm

class gv:
    face = []
    result = 0

window = tk.Tk()
window.title('Face Recognitoin')
window.geometry('600x400')

def traning():
    eigenface.traningdata()
    tm.showinfo('Info', 'Training is done!')

def Recognition():
    gv.result = eigenface.Recognition(gv.face, eigenface.globalvalue.component_set_gv, eigenface.globalvalue.k_gv, eigenface.globalvalue.eigenface_mat_gv, eigenface.globalvalue.sample_num_gv)
    tm.showinfo('Info', 'This Face is ' + str(eigenface.globalvalue.map_gv[gv.result]) )

def choosingface():
    file = fd.LoadFileDialog(window)
    filename = file.go('/Users/xavier0121/Desktop/test_recog/orl_faces/')
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im_double = eigenface.im2double(image)
    im_v = eigenface.img2vector(im_double)
    gv.face = im_v - eigenface.globalvalue.mean_gv

def loadingnewfaces():
    filename = tfd.askdirectory()
    eigenface.loadingfaces(filename+'/')


b1 = tk.Button( window, text = 'Training DataSet', width = 15, height = 5, command = traning )
b1.place(x = 100, y = 100)

b2 = tk.Button( window, text = 'Recognition', width = 15, height = 5, command = Recognition )
b2.place( x = 300, y = 200 )

b3 = tk.Button( window, text = 'Choose a face', width = 15, height = 5, command = choosingface )
b3.place( x = 300, y = 100 )

b4 = tk.Button( window, text = 'loading new faces', width = 15, height = 5, command = loadingnewfaces )
b4.place( x = 100, y = 200 )



window.mainloop()