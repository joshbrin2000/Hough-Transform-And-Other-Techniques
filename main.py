# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
from PIL import ImageFilter
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def houghTransLine():
        global test
        global imageOrig
        global panelA, panelB
        global saveIm
        global imSwitch
        global threshold

        copy = imageOrig.copy()

        gray = cv2.cvtColor(imageOrig, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

        lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
        #print(lines)
        if lines is not None:
                for rThet in lines:
                    array = np.array(rThet[0], dtype=np.float64)
                    r, theta = array
                    a = np.cos(theta)
                    b = np.sin(theta)
                 
                    x_1 = int(a*r + 1000 * (-b))
                    y_1 = int(b*r + 1000 * a)
                    x_2 = int(a*r - 1000 * (-b))
                    y_2 = int(b*r - 1000 * a)
                    
                    cv2.line(copy, (x_1, y_1), (x_2, y_2), (0, 185, 255), 2)

                image = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
                Im = Image.fromarray(image)
                Im = ImageTk.PhotoImage(Im)
                panelB.configure(image=Im)
                panelB.image = Im
        else:
                print("No Lines Detected")

def houghTransCircle():
        global test
        global imageOrig
        global panelA, panelB
        global saveIm
        global imSwitch
        global threshold
        
        copy = imageOrig.copy()
        
        gray = cv2.cvtColor(imageOrig, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 75)
        
        if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (a, b, c) in circles:
                        cv2.circle(copy, (a, b), c, (0, 185, 255), 4)

        image = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
        Im = Image.fromarray(image)
        Im = ImageTk.PhotoImage(Im)
        panelB.configure(image=Im)
        panelB.image = Im

def addNoise():
        global test
        global imageOrig
        global panelA, panelB
        global saveIm
        global imSwitch
        global threshold
        global filtLen
        
        #cv2.imshow('aa', imageOrig)
        row,col,ch = imageOrig.shape
        mean = 0
        var = 0.1
        sigma = var**0.1
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch).astype('uint8')
        noisy = cv2.add(imageOrig, gauss)
        imageOrig = noisy
        #cv2.imshow('aa', imageOrig)
        #cv2.imshow('aa', noisy)
        noisyDisp = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        #cv2.imshow('aa', noisyDisp)
        noisyDisp = Image.fromarray(noisyDisp)
        noisyDisp = ImageTk.PhotoImage(noisyDisp)
        panelA.configure(image=noisyDisp)
        panelA.image = noisyDisp

def fourierTrans():
        global test
        global imageOrig
        global panelA, panelB
        global saveIm
        global imSwitch
        global threshold
        global filtLen

        gray = cv2.cvtColor(imageOrig,cv2.COLOR_BGR2GRAY)
        img = np.float32(gray)

        dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
        dftShift = np.fft.fftshift(dft)

        result = 20*np.log(cv2.magnitude(dftShift[:,:,0], dftShift[:,:,1]))
        plt.imshow(result, cmap = 'gray')
        plt.title('Fourier Transform of Image')
        plt.show()
        
        rows, columns = gray.shape
        cRows = int(rows/2)
        cColumns = int(columns/2)

        matrix = np.zeros((rows, columns, 2), np.uint8)
        matrix[cRows-30:cRows+30, cColumns-30:cColumns+30] = 1

        fShift = dftShift * matrix
        fInverseShift = np.fft.ifftshift(fShift)
        result = cv2.idft(fInverseShift)
        result = cv2.magnitude(result[:,:,0], result[:,:,1])
        plt.imshow(result)
        plt.show()

def canny():
        global test
        global imageOrig
        global panelA, panelB
        global saveIm
        global imSwitch
        global threshold
        global filtLen

        gray = cv2.cvtColor(imageOrig,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        image = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)
        Im = Image.fromarray(image)
        Im = ImageTk.PhotoImage(Im)
        panelB.configure(image=Im)
        panelB.image = Im

def select_image():
	global panelA, panelB
	global en1
	global imageOrig
	global filtLen
	global threshold
	global offset
	global saveIm
	global imSwitch
	threshold = 127
	offset = 1
	path = filedialog.askopenfilename()
	if len(path) > 0:
		imageOrig = cv2.imread(path)
		#.astype(np.uint8)
		filtLen = 5
		# OpenCV represents images in BGR order; however PIL represents
		# images in RGB order, so we need to swap the channels
		image = cv2.cvtColor(imageOrig, cv2.COLOR_BGR2RGB)
		# convert the images to PIL format...
		image = Image.fromarray(image)
		image = ImageTk.PhotoImage(image)
		
		if panelA is None or panelB is None:
			# the first panel will store our original image
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="left", padx=10, pady=10)
			# while the second panel will store the edge map
			panelB = Label(image=image)
			panelB.image = image
			panelB.pack(side="right", padx=10, pady=10)

			saveIm = imageOrig
			imSwitch = imageOrig
			
			btn2 = Button(root, text="Hough Transform: Line", command=houghTransLine)
			btn2.pack(side="bottom", padx="10", pady="10")

			btn3 = Button(root, text="Hough Transform: Circle", command=houghTransCircle)
			btn3.pack(side="bottom", padx="10", pady="10")

			btn4 = Button(root, text="Fourier Transform", command=fourierTrans)
			btn4.pack(side="bottom", padx="10", pady="10")

			btn5 = Button(root, text="Canny Edge Detection", command=canny)
			btn5.pack(side="bottom", padx="10", pady="10")

		# otherwise, update the image panels
		else:
			# update the pannels
			panelA.configure(image=image)
			panelB.configure(image=image)
			panelA.image = image
			panelB.image = image

root = tk.Tk("Hought Transform")
panelA = None
panelB = None
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn1 = Button(root, text="Select an image", command=select_image)
btn1.pack(side="bottom", padx="10", pady="10")

root.mainloop()
