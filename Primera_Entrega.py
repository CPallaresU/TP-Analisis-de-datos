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


for q in range(0,10):
    
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
        

    
    print("Número: {}".format(q))
    
    ##CREACION DATASETS
    col = np.arange(28*28)
    df = pd.DataFrame(lista_img,columns=col)
    """
    component_v=[] #No el de Homelander
    df_v = pd.DataFrame()
    k_iter  = 5
    c_iter = 5
    
    df = pd.read_csv(str(q)+".csv")
    
    
    #GENERO PCA PARA UN NUMERO CON UN MAXIMO DE 25 COMPOENTES QUE EXPLIQUEN A PARTIR DEL DATAFRAME DF
    
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
        
        
         #x = df_v[:-1].loc[(df_v!=0.0).any(axis=1)].index
         
         
         
    #DF_V TIENE 784 FILAS Y N COLUMNAS COMO COMPONENTES DECIDÍ EVALUAR, BUSCO SI A MEDIDA QUE AUMENTO EL NÚMERO
    #DE COMPONENTES QUE EXPLICAN EN AL MENOS UN 90%, ESE APORTE POR PIXEL AUMENTA O NO.
    
    df_pca = pd.Series(df_v[df_v.columns[-1]] - df_v[df_v.columns[0]]).to_frame() #ESTOY BUSCANDO CUALES COLUMNAS CAMBIARON O APORTARON ENTRE LA PRIMER Y ULTIMA PCA
    
    indexes_to_delete = list(df_pca[:-1].loc[(df_pca[:-1] == 0.0 ).any(axis=1)].index)  # Indices de columnas que no aportan informacion
    
    df_t = df.drop(indexes_to_delete , axis = 1)
    
    df_t.to_csv(str(q)+".csv",index=False) # Guardando dataset sin esas columnas
    
    
    #DESDE LA LINEA 37 HASTA LA 110 GENERAMOS .CSV POR CADA NÚMERO SIN ESAS COLUMNAS QUE NO APORTARON AL PCA
    
    print(len(df_t.columns))
    
    
    #SE GRAFICA PCA DE CADA DATASET .CSV
    
    for r in range(0,14):
        
        start = r*56
        end = (r+1)*56        
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
    

#GENERO DATASET CON NANS A PARTIR DE MI DATASET SIN CIERTAS COLUMNAS
    
for i in range (10):
    dt.dirtydata(str(i)+".csv") #CREACION DE DATASET DIRT CON NAN


############
############
### MICE ###
############
############




for i in range(10):
    
    if i == 1:
    
        print(i)
        df_missing = pd.read_csv("dirt_"+str(i)+".csv").head(500)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        df_0 = pd.DataFrame(imp.fit_transform(df_missing),columns = df_missing.columns)/256 # y = b_1*x1 + b_2*x2 + ... +b_n*xn + t
                                                                                            # genero un nuevo dataset ajustado bajo la regresion
                                                                                            # de unos datos divididos por 256, y puedo volver
                                                                                            # a multiplicarlo por 256 ya que:
                                                                                            # y*256 = b_1*(x1*256) + b_2*(x2*256) + ... +b_n*(xn*256) + t*(256)
        df_00 = df_0.copy()
        err = 0
    
        sw = 0
        while sw == 0 : #for e in range(100) :
            
            acum = 0
            for k in df_missing.columns:
            
                col_to_impute = str(k)
                df_2 = df_0[df_missing[col_to_impute].notna()] #With no nan in certain column
                df_1 = df_0[df_missing[col_to_impute].isna()]#With no nan in certain column
                features_columns = df_missing.columns[df_missing.columns!=str(k)]
                t,l = mice.impute_column(df_1,df_2,str(k),features_columns) #l : mis indices y t: los valores nuevos en los nan
                acum = acum + sum(abs(df_0[str(k)][l] - t))
                df_0[str(k)][l]=t #Asignación nueva con datos inferidos
        
        
            err = acum / len(df_missing[col_to_impute].isna())
            print(err)
            df_00 = df_0.copy() #Es la nueva matriz
            
            if err <= 0.08 :
                df_00.to_csv(str(i)+"_mice.csv",index=False)
                sw=1;













