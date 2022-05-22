# -*- coding: utf-8 -*-
"""
Created on Fri May 13 17:25:56 2022

@author: 10
"""
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import data_dirt as dt
import MICE as mice
from sklearn.impute import SimpleImputer




def lista_png (val):
    
    filelist=os.listdir(val)
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")):
            filelist.remove(fichier)
    return filelist


for q in range(0,1):
    
    """
    lista_img=[]

    t=lista_png(str(q))
    
    ###LECTURA DE TODAS LAS IMAGENES
    
    for j in t:
        result= cv2.imread(str(q)+"/"+j)
        img_=[]

        for x in range(0,28):
            
            for y in range (0,28):
                img_.append(np.sum(result[x][y])/3) 
        
        
        
        #gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        #gray_image=list(gray_image.reshape(784,1))
        #lista_img.append(gray_image)
        lista_img.append(img_)
    """    
    component_v=[] #No el de Homelander
    df_v = pd.DataFrame()
    k_iter  = 5
    c_iter = 5
    
    print("Número: {}".format(q))
    
    ##CREACION DATASETS
    """
    col = np.arange(28*28)
    df = pd.DataFrame(lista_img,columns=col)
    df.to_csv(str(q)+".csv",index=False)
    """
    df = pd.read_csv(str(q)+".csv")
    
    for k in range(1,k_iter):
        
        n_= c_iter * k
        pca = PCA(n_components=n_)
        pca.fit_transform(df);
        
        v=pca.singular_values_
        l=[]
        sw=True
        
        for i in range(0,n_):
            
          value=sum(v[0:i])/sum(v)
          
          l.append(value)
          
          if value > .9 and sw:
            sw=False
            print("Son suficiente {} componentes para explicar los 784 atributos con un {}%".format(i,value*100))
            break
        
        index_= [i for i in range(0,n_)]
        df_componentes_pca=pd.DataFrame(pca.components_,columns=df.columns,index = index_)
        
        columnas = list(df.columns)
        datos = list(sum(np.abs(pca.components_)))
        str_="Correlación por atributo para {} componentes".format(n_)
        df_corr=pd.DataFrame(data = datos,index=columnas,columns=[str_])
        df_corr.style.set_properties(**{'text-align': 'center'})
        df_v["Componentes PCA:"+str(n_)] = df_corr[str_]
        
    for r in range(0,1):
        
        start = r*50
        end = (r+1)*50
        df_v.iloc[start:end].plot(title="Número: "+str(q)+ " - Segmento: {} ".format(r))
        """
        x = np.arange(k_iter-1)
        y = df_v.iloc[start:end].index 
        X,Y = np.meshgrid(x,y)
        Z =df_v.iloc[start:end]
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.locator_params ('z', nbins = 10)
        ax.scatter(Y, X, Z)
        #ax.view_init(0, 90)
        ax.set_title("Número: {}".format(q))
        ax.set_ylabel('N-COMPONENTES')
        ax.set_xlabel('RANGO DE PIXELES')
        ax.set_zlabel('PESO APORTADO PIXEL')
        """
    #ax.scatter(Y, X, Z)
    
   
for i in range (1):
    dt.dirtydata(str(i)+".csv")
  
 

for i in range(1):
    
    df_missing = pd.read_csv("dirt_"+str(i)+".csv")
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_0 = pd.DataFrame(imp.fit_transform(df_missing),columns = df_missing.columns)
    df_00 = df_0.copy()
    err = 0


    for e in range(100) :
        
        for k in range(350,352):
        
            col_to_impute = str(k)
            df_2 = df_0[df_missing[col_to_impute].notna()] #With no nan in certain column
            df_1 = df_0[df_missing[col_to_impute].isna()]#With no nan in certain column
            features_columns = df_missing.columns[df_missing.columns!=str(k)]
            t,i = mice.impute_column(df_1,df_2,str(k),features_columns) #i : mis indices y t: los valores nuevos en los nan
            df_0[str(k)][i]=t #Asignación nueva con datos inferidos
    
    
        val = (df_0 - df_00).values
        er = np.power(val,2)
        err = np.sum(er)/len(er)
        print(err)
        df_00 = df_0.copy() #Es la nueva matriz
        
        if err < 0.1 :
            break;





















