# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 01:49:48 2022

@author: 10
"""

import pandas as pd
import math
import statistics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from PIL import Image


def fila(x):
    return math.ceil(x/28)-1

def columna(x):
    return (x%28)-1



c1=[]
c2=[]
c3=[]
c4=[]
c5=[]
c6=[]
c7=[]
c8=[]
c9=[]
c10=[]
c11=[]
c12=[]
c13=[]
c14=[]
c15=[]
c16=[]

cont=0


df = pd.read_csv("5_mice.csv")


for i in range(1):
    
    #df = pd.read_csv("9_mice.csv")

    for k in df.columns:
        
        sw=0
        
        
        if fila(int(k)) <= 7  and columna(int(k))  <= 7:
            sw=1
            cont=cont+1
            c1.append(int(k))
        
        if fila(int(k)) <= 7 and columna(int(k))  > 7 and columna(int(k))  <= 14:
            sw=1
            cont=cont+1
            c2.append(int(k))
            
        if fila(int(k))  <= 7 and columna(int(k))  > 14 and columna(int(k))  <= 21:
            sw=1
            cont=cont+1
            c3.append(int(k))      
            
        if fila(int(k))  <= 7 and columna(int(k))  > 21 and columna(int(k))  <= 28:
            sw=1
            cont=cont+1
            c4.append(int(k))    
        
            
            
        
        if fila(int(k))  > 7 and fila(int(k))  <= 14 and columna(int(k))  <= 7 :
            c5.append(int(k))
            cont=cont+1
            sw=1
            
        if fila(int(k))  > 7 and fila(int(k))  <= 14 and columna(int(k))  > 7 and columna(int(k))  <= 14:
            c6.append(int(k))
            cont=cont+1
            sw=1
        
        if fila(int(k))  > 7 and fila(int(k))  <= 14 and columna(int(k))  > 14 and columna(int(k))  <= 21:
            c7.append(int(k))
            cont=cont+1
            sw=1
        
        if fila(int(k))  > 7 and fila(int(k))  <= 14 and columna(int(k))  > 21 and columna(int(k))  <= 28:
            c8.append(int(k))
            cont=cont+1
            sw=1
            
            
            
        
        if fila(int(k))  > 14 and fila(int(k))  <= 21 and columna(int(k))  <= 7:
            c9.append(int(k))
            cont=cont+1
            sw=1
        
        if fila(int(k))  > 14 and fila(int(k))  <= 21 and columna(int(k))  > 7 and columna(int(k))  <= 14:
            c10.append(int(k))
            cont=cont+1
            sw=1
        
        if fila(int(k))  > 14 and fila(int(k))  <= 21 and columna(int(k))  > 14 and columna(int(k))  <= 21:
            c11.append(int(k))
            cont=cont+1
            sw=1
        
        if fila(int(k))  > 14 and fila(int(k))  <= 21 and columna(int(k))  > 21 and columna(int(k))  <= 28:
            c12.append(int(k))
            cont=cont+1
            sw=1
            
            
        
        if fila(int(k))  > 21 and fila(int(k))  <= 28 and columna(int(k))  <= 7:
            c13.append(int(k))
            cont=cont+1
            sw=1
        
        if fila(int(k))  > 21 and fila(int(k))  <= 28 and columna(int(k))  > 7 and columna(int(k))  <= 14:
            c14.append(int(k))
            cont=cont+1
            sw=1
        
        if fila(int(k))  > 21 and fila(int(k))  <= 28 and columna(int(k))  > 14 and columna(int(k))  <= 21:
            c15.append(int(k))
            cont=cont+1
            sw=1
        
        if fila(int(k))  > 21 and fila(int(k))  <= 28 and columna(int(k))  > 21 and columna(int(k))  <= 28:
            c16.append(int(k))
            cont=cont+1
            sw=1
            

        if sw == 0:
            print(int(k))

        
dict_ = {"1": c1 , "2": c2, "3": c3, "4": c4, "5": c5, "6": c6, "7": c7,"8": c8
         ,"9": c9, "10": c10, "11": c11, "12": c12,"13": c13 , "14": c14, "15": c15 , "16": c16}
            

lista_mean = []
lista_std=[]
list_std=[]
for i in range(1,17):
   acum_mean = 0
   acum_std = 0
   for j in dict_.get(str(i)):
       acum_mean = acum_mean + df[str(j)].mean()
       lista_std.append(df[str(j)].std())
       
   if len(dict_.get(str(i))) == 1:
       list_std.append(dict_.get(str(i))[0])
   else:    
       list_std.append(statistics.stdev(lista_std))
   lista_std=[]
   lista_mean.append(acum_mean / len(dict_.get(str(i)))) #Lista de todos las medias por cuadrante, son 16 posiciones solamente.
   
   
list_img = []


df["C1"] = 500*[0]
df["C2"] = 500*[0]
df["C3"] = 500*[0]
df["C4"] = 500*[0]
df["C5"] = 500*[0]
df["C6"] = 500*[0]
df["C7"] = 500*[0]
df["C8"] = 500*[0]
df["C9"] = 500*[0]
df["C10"] = 500*[0]
df["C11"] = 500*[0]
df["C12"] = 500*[0]
df["C13"] = 500*[0]
df["C14"] = 500*[0]
df["C15"] = 500*[0]
df["C16"] = 500*[0]



for i in range(0,len(df)):
    
   means_ = []

   for j in range(1,17):
       
       str_c = "C"+str(j)
       acum_ = 0
       for k in dict_.get(str(j)): #Calculo de media por cuadrante a partir de la lista ya calculada de medias
           
           acum_ = acum_ + df[str(k)][i]            
       
       df[str_c][i] = acum_  / len(dict_.get(str(j)))
       means_.append(acum_  / len(dict_.get(str(j)))) ##Lista de media por cuadrante
   
   check_mean_per_q = []
   list_img.append(mean_squared_error(lista_mean,means_)) #Que tan cercanos están a los valores de lista_mean
   
   #for n in range(16):
       #check_mean_per_q.append(((means_[n] - lista_mean[n])*(means_[n] - lista_mean[n]))/ 2)
   #print(check_mean_per_q)   



#plt.plot(list_img,list(np.arange(500)))
## El eje X me muestra el error promedio de cada imagen con respecto al
## valor medio de cada cuadrante original vs el calculado de cada imagen


#row = df.iloc[0]
#row.plot(kind='bar')

cont = 0
print(statistics.mean(list_img) + statistics.stdev(list_img))
outliers = []
not_outliers = []
for k in range(500):
    if list_img[k] < statistics.mean(list_img) + statistics.stdev(list_img) :
        not_outliers.append(k)
        cont=cont+1
    else:
        outliers.append(k)
    
print(cont)  
  

"""

df["PROMEDIO"] = list_img
df.iloc[outliers].plot(y=["C1","C2","C3","C5","C6","C7","C8","C9"])
df.iloc[not_outliers].tail(len(outliers)).plot(y=["C1","C2","C3","C5","C6","C7","C8","C9"])

plt.plot(df.iloc[not_outliers].tail(len(outliers))["PROMEDIO"],outliers)

plt.plot(df.iloc[outliers]["PROMEDIO"],outliers)

"""    

#GRÁFICO DE DENDISDAD


dq = pd.read_csv("9_mice.csv")        

for l in outliers:    
    row = dq.iloc[0]
    row.plot(kind='density',color = "y")

dq = pd.read_csv("8_mice.csv")        

for l in outliers:    
    row = dq.iloc[0]
    row.plot(kind='density',color = "b")
    


#RELLENAR DATASET PARA RECONSTRUCCIÓN DE LAS IMAGENES NORMALES Y DE OUTLIERS



def imagenes_reconstruct(idx_img, img_, dir_,y):
            
    lista_ = [0]*784
    cont = 0
    for k in idx_img:
        if any(c.isalpha() for c in k) == False:
            lista_[int(k)] = int(abs(img_[cont]*256))
            cont = cont + 1
    
    
    pixels = lista_
    
    # Use PIL to create an image from the new array of pixels
    
    
    image_out = Image.new(mode="1",size=(28,28))
    image_out.putdata(pixels)
    
    image_out.save(dir_+"/"+str(y)+'.png')


for y in range(len(outliers)):

    idx_img=list(df.iloc[outliers[y]].index)
    img_ = list(df.iloc[outliers[y]])
    imagenes_reconstruct(idx_img,img_,"outliers",y)


for w in range(len(df)):
    
    if w not in outliers:
        idx_img=list(df.iloc[w].index)
        img_ = list(df.iloc[w])
        imagenes_reconstruct(idx_img,img_,"normal",w)

          
