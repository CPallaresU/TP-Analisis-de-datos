# -*- coding: utf-8 -*-
"""data besmircher.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G7OhzgWhlfZozXLRlDV7luidIlYi2PLy

# Data Masker
"""

L=enumerate
E=None
C=print
D=len
import pandas as G,numpy as A,requests as H
import pandas as pd

def generate_nulls(df,cols=E,percentage=0.05):
	C=cols;B=df;C=B.columns.tolist()if C is E else C;G=A.random.randint(100,size=D(C));H=A.math.floor(D(B)*percentage);
	for (I,F) in L(C):A.random.seed(G[I]);J=B[F].sample(n=H).index;B[F].iloc[J]=A.nan if B[F].dtype in[int,float]else E
	return B
def sparsify_data(df,percentage=0.02):
	B=df;F=A.random.randint(100,size=D(B));G=A.math.floor(D(B)*percentage);H=B.sample(n=G).index
	for (I,C) in L(H):A.random.seed(F[I]);B.iloc[C]={D:F if A.random.binomial(1,0.5)else E for(D,F)in B.iloc[C].items()}


def dirtydata(name_):

    name = name_
    input_df = pd.read_csv(name)
    
    cols = [
            ## AQUI PONER LAS COLUMNAS QUE QUIEREN MODIFICAR
    ]
    
    output_df = generate_nulls(input_df, cols or None, 0.15)
    
    output_df.to_csv("dirt_"+name, index = False)

