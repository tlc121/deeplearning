import Tkinter as tk
import face_detector as fd
import faceSwap as fw
import face_average as fa
import triangulation as tri
import morph_Addpoints as add
import face_morphing as fm
import video
import os






    
def insertimage():
    fd.set_image_name(insert_image.get())
    fd.set_part(var.get())
    fd.main()

def Exit():
    exit()
    

def facedetector():
    fd.set_image_name(insert_image.get())
    fd.set_part(var.get())
    fd.main()
    fd.show()
    
    
def faceswap():
    fw.set_image_name(image_name1.get(),image_name2.get())
    fw.set_part(var.get())
    fw.main()

def faceaverage():
    fa.set_image_name(image_name2.get())
    fa.set_part(var.get())
    fa.main()

def triangulation():
    add.average()
    tri.main()

def result():
    fm.set_image_name(image_name2.get())
    fm.main()
    video.main()
    video.playvideo()
    #os.execfile("/Users/xavier0121/Desktop/saveVideo.avi")
    
def delay(): 
    print var.get()
    label.config(text = 'You have selected ' + var.get() )
        
window = tk.Tk()
window.title('Expression Morphing!!')
window.geometry('800x600')

var = tk.StringVar()
var1 = tk.StringVar()


b1 = tk.Button( window, text = 'Insert', width = 15, height = 5, command = insertimage )
b1.place(x = 145, y = 15)

b2 = tk.Button( window, text = 'Swap!', width = 15, height = 5, command = faceswap )
b2.place( x = 275, y = 115 + 150 )

b3 = tk.Button( window, text = 'Landmarks', width = 15, height = 5, command = facedetector )
b3.place( x = 350, y = 15)

b4 = tk.Button( window, text = 'Average', width = 15, height = 5, command = faceaverage )
b4.place( x = 10, y = 200 + 150)

b5 = tk.Button( window, text = 'Generate Delaunay Triangulation', width = 15, height = 5, command = triangulation)
b5.place( x = 275, y = 200 + 150)

b6 = tk.Button( window, text = 'Exit', width = 15, height = 5, command = Exit )
b6.place( x = 550, y = 15 )

b7 = tk.Button( window, text = 'Boom!', width = 15, height = 5, command = result)
b7.place( x = 550, y = 200 + 150)

r1 = tk.Radiobutton( window , text = 'mouth', variable = var, value = 'mouth', command = delay )
r1.place( x =10, y = 100 )

r2 = tk.Radiobutton( window , text = 'eyes', variable = var, value = 'eyes', command = delay )
r2.place( x =10, y = 130 )

r3 = tk.Radiobutton( window , text = 'eyebrow', variable = var, value = 'eyebrow', command = delay )
r3.place( x =10, y = 160 )

r4 = tk.Radiobutton( window , text = 'above all', variable = var, value = 'above all', command = delay )
r4.place( x =10, y = 190 )

label = tk.Label(window, bg = 'yellow', text = 'Please choose a part of face you wanna change', width = 40, height = 5)
label.place( x = 200, y = 115 )

insert_image = tk.Entry(window)
insert_image.place( x = 130, y = 10 )

tk.Label(window, text = 'Insert the image: ').place(x = 10, y = 10)
tk.Label(window, text = 'Swap Image from: ' ).place( x = 10, y = 100 + 150 )

image_name1 = tk.Entry(window)
image_name1.place(x = 130, y = 98 + 150 )

tk.Label(window, text = 'to' ).place( x = 350 , y = 100 + 150 )

image_name2 = tk.Entry(window)
image_name2.place(x = 400, y = 100 + 150 )




window.mainloop()


