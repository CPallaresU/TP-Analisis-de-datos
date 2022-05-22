# -*- coding: utf-8 -*-
"""
Created on Sat May 21 20:01:39 2022

@author: 10
"""

from sklearn.linear_model import LinearRegression


def impute_column(df_nan,df_no_nan, col_to_predict, feature_columns):

  X_train,y_train = df_no_nan[feature_columns] , df_no_nan[col_to_predict]
  X_pred = df_nan[feature_columns]

  model = LinearRegression()

  model.fit(X_train,y_train)
  df = model.predict(X_pred)
  
  return df, X_pred.index





















