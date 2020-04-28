# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:35:27 2020

@author: gpith
"""

from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab
import numpy as np
from keras.models import load_model
import cv2

def image_predict(image_name):
    #READING THE IMAGE    
    image=cv2.imread(image_name+".jpg")
    #CONVERTING THE IMAGE TO GRAYSCALE
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #INVERTING THE COLORS AND RESIZING
    image = cv2.resize(255-image, (28, 28))
    #THRESHOLDING
    (thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #NORMALIZING THE IMAGE
    image=image/255
    #REMOVING THE EXTRA PADDING SURROUNDING THE DIGIT
    while np.sum(image[0]) == 0: #EXTRA PADDING FROM TOP
        image = image[1:]

    while np.sum(image[:,0]) == 0: #EXTRA PADDING FROM LEFT
        image = np.delete(image,0,1)
    
    while np.sum(image[-1]) == 0: #EXTRA PADDING FROM BOTTOM
        image = image[:-1]
    
    while np.sum(image[:,-1]) == 0: #EXTRA PADDING FROM RIGHT
        image = np.delete(image,-1,1)
    
    #SCALING THE IMAGE TO 20*20 MAINTAINING THE ASPECT RATIO
    rows,cols = image.shape
    factor=20.0/max((rows,cols)) 
    rows=int(rows*factor)
    cols=int(cols*factor)
    image=cv2.resize(image,(int(cols),int(rows)))
    
    #PADDING THE IMAGE AROUND TO RESIZE IT TO 28*28
    shiftx=28-cols
    shifty=28-rows
    top, bottom = shifty//2,shifty-(shifty//2)
    left, right = shiftx//2,shiftx-(shiftx//2)
    color = [0, 0, 0]
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    
    #LOADING THE PRETRAINED MODEL
    model=load_model("mnistCNN")
    image=image.reshape(1,28,28,1)
    prediction=np.argmax(model.predict(image))
    print("Digit is: ",prediction)
    
class Predictor(tk.Tk):
    
    def predict(self,image):
        image=np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(255-image, (28, 28))
        
        image=image/255
        image=image.reshape(1,28,28,1)
        model=load_model("mnistCNN")
        prediction=np.argmax(model.predict(image))
        return prediction
    
    def __init__(self):
            tk.Tk.__init__(self)
            self.x = self.y = 0
            self.canvas=tk.Canvas(self, width=200, height=200, bg='white',cursor='cross')
            self.label=tk.Label(self,text="0",font=("Helvetica",48))
            self.classify=tk.Button(self,text="Recognize",command=self.classify)
            self.clear=tk.Button(self,text="Clear",command=self.clear)
            self.canvas.grid(row=0, column=0, pady=2)
            self.label.grid(row=0, column=1,pady=2, padx=2)
            self.classify.grid(row=1, column=0, pady=2, padx=2)
            self.clear.grid(row=1, column=1, pady=2)
        
            self.canvas.bind("<B1-Motion>",self.draw)
    
    def clear(self):
        self.canvas.delete("all")
        self.label.configure(text=str(0))
        
    def draw(self,event):
        self.x=event.x
        self.y=event.y
        r=5
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill="black")
        
    def classify(self):
        canvas_info=self.canvas.winfo_id()
        frame=win32gui.GetWindowRect(canvas_info)
        image=ImageGrab.grab(frame)
        digit=self.predict(image)
        self.label.configure(text=str(digit))
        
# =============================================================================
# app=Predictor()
# mainloop()
# =============================================================================
image_predict("2")

        
            
    
        