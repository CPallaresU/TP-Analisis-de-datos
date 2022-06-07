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

for i in range(1):
    
    df = pd.read_csv("9_mice.csv")

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
       
   list_std.append(statistics.stdev(lista_std))
   lista_std=[]
   lista_mean.append(acum_mean / len(dict_.get(str(i)))) #Lista de todos las medias por cuadrante de imagen
   
   
list_img = []

for i in range(0,len(df)):
    
   means_ = []

   for j in range(1,17):
       acum_ = 0
       for k in dict_.get(str(j)): #Calculo de media por cuadrante a partir de la lista ya calculada de medias
           
           acum_ = acum_ + df[str(k)][i]            

       means_.append(acum_  / len(dict_.get(str(j)))) ##Lista de media por cuadrante
   
   check_mean_per_q = []
   list_img.append(mean_squared_error(lista_mean,means_))
   
   #for n in range(16):
       #check_mean_per_q.append(((means_[n] - lista_mean[n])*(means_[n] - lista_mean[n]))/ 2)
   #print(check_mean_per_q)   



plt.plot(list_img,list(np.arange(500)))
## El eje X me muestra el error promedio de cada imagen con respecto al
## valor medio de cada cuadrante original vs el calculado de cada imagen


#row = df.iloc[0]
#row.plot(kind='bar')

cont = 0
print(statistics.mean(list_img) + statistics.stdev(list_img))
for k in range(500):
    if list_img[k] < statistics.mean(list_img) + statistics.stdev(list_img) :
        cont=cont+1
print(cont)    
            
            
            
            
            
            
            
            
