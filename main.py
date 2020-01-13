from realnowcast import *
from visualize import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings
import pickle

warnings.simplefilter('ignore')
###############################################################################################
# 15 days before nowcasting
#########################################
#### Pseudo setting
## pseudo dataset
pseudo_dataset15 = Dataset(setting='pseudo', predict_period=30, valid_size=12, days_before=15)
pseudo_dataset15.build()

## ar1
pseudo_ar115 = Model(method='ar1', dataset=pseudo_dataset15)
pseudo_ar115.execute()
    
## lin reg
pseudo_lin_reg15 = Model(method='lin_reg', dataset=pseudo_dataset15)
pseudo_lin_reg15.execute()

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
pseudo_lasso15 = Model(method='lasso', dataset=pseudo_dataset15)
pseudo_lasso15.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(pseudo_lasso15)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
pseudo_ridge15 = Model(method='ridge', dataset=pseudo_dataset15)
pseudo_ridge15.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(pseudo_ridge15)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
pseudo_elastic15 = Model(method='elastic', dataset=pseudo_dataset15)
pseudo_elastic15.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(pseudo_elastic15.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(pseudo_elastic15, i)
    
## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
pseudo_rf15 = Model(method='rf', dataset=pseudo_dataset15)
pseudo_rf15.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(pseudo_rf15.predict_period):
    adviz_X_Y_Zrmse(pseudo_rf15, i)

#########################################
#### fully setting pseudo validation

## full pseudo validation dataset
full_pseudo_dataset15 = Dataset(setting='full', validation='pseudo', predict_period=30, valid_size=12, days_before=15)
full_pseudo_dataset15.build()

## ar1
full_pseudo_ar115 = Model(method='ar1', dataset=full_pseudo_dataset15)
full_pseudo_ar115.execute()

## lin reg
full_pseudo_lin_reg15 = Model(method='lin_reg', dataset=full_pseudo_dataset15)
full_pseudo_lin_reg15.execute()

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
full_pseudo_lasso15 = Model(method='lasso', dataset=full_pseudo_dataset15)
full_pseudo_lasso15.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_pseudo_lasso15)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
full_pseudo_ridge15 = Model(method='ridge', dataset=full_pseudo_dataset15)
full_pseudo_ridge15.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_pseudo_ridge15)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
full_pseudo_elastic15 = Model(method='elastic', dataset=full_pseudo_dataset15)
full_pseudo_elastic15.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_pseudo_elastic15.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(full_pseudo_elastic15, i)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
full_pseudo_rf15 = Model(method='rf', dataset=full_pseudo_dataset15)
full_pseudo_rf15.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_pseudo_rf15.predict_period):
    adviz_X_Y_Zrmse(full_pseudo_rf15, i)
    
## full full validation dataset
full_full_dataset15 = Dataset(setting='full', validation='full', predict_period=30, valid_size=12, days_before=15)
full_full_dataset15.build()

## ar1
full_full_ar115 = Model(method='ar1', dataset=full_full_dataset15)
full_full_ar115.execute()

## lin reg
full_full_lin_reg15 = Model(method='lin_reg', dataset=full_full_dataset15)
full_full_lin_reg15.execute()

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
full_full_lasso15 = Model(method='lasso', dataset=full_full_dataset15)
full_full_lasso15.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_full_lasso15)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
full_full_ridge15 = Model(method='ridge', dataset=full_full_dataset15)
full_full_ridge15.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_full_ridge15)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
full_full_elastic15 = Model(method='elastic', dataset=full_full_dataset15)
full_full_elastic15.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_full_elastic15.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(full_full_elastic15, i)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
full_full_rf15 = Model(method='rf', dataset=full_full_dataset15)
full_full_rf15.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_full_rf15.predict_period):
    adviz_X_Y_Zrmse(full_full_rf15, i)

###############################################################################################
# 45 days before nowcasting
#########################################
#### Pseudo setting
## pseudo dataset
pseudo_dataset45 = Dataset(setting='pseudo', predict_period=30, valid_size=12, days_before=45)
pseudo_dataset45.build()

## ar1
pseudo_ar145 = Model(method='ar1', dataset=pseudo_dataset45)
pseudo_ar145.execute()

## lin reg
pseudo_lin_reg45 = Model(method='lin_reg', dataset=pseudo_dataset45)
pseudo_lin_reg45.execute()

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
pseudo_lasso45 = Model(method='lasso', dataset=pseudo_dataset45)
pseudo_lasso45.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(pseudo_lasso45)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
pseudo_ridge45 = Model(method='ridge', dataset=pseudo_dataset45)
pseudo_ridge45.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(pseudo_ridge45)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
pseudo_elastic45 = Model(method='elastic', dataset=pseudo_dataset45)
pseudo_elastic45.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(pseudo_elastic45.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(pseudo_elastic45, i)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
pseudo_rf45 = Model(method='rf', dataset=pseudo_dataset45)
pseudo_rf45.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(pseudo_rf45.predict_period):
    adviz_X_Y_Zrmse(pseudo_rf45, i)

#########################################
#### fully setting pseudo validation

## full pseudo validation dataset
full_pseudo_dataset45 = Dataset(setting='full', validation='pseudo', predict_period=30, valid_size=12, days_before=45)
full_pseudo_dataset45.build()
    
## ar1
full_pseudo_ar145 = Model(method='ar1', dataset=full_pseudo_dataset45)
full_pseudo_ar145.execute()

## lin reg
full_pseudo_lin_reg45 = Model(method='lin_reg', dataset=full_pseudo_dataset45)
full_pseudo_lin_reg45.execute()

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
full_pseudo_lasso45 = Model(method='lasso', dataset=full_pseudo_dataset45)
full_pseudo_lasso45.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_pseudo_lasso45)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
full_pseudo_ridge45 = Model(method='ridge', dataset=full_pseudo_dataset45)
full_pseudo_ridge45.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_pseudo_ridge45)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
full_pseudo_elastic45 = Model(method='elastic', dataset=full_pseudo_dataset45)
full_pseudo_elastic45.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_pseudo_elastic45.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(full_pseudo_elastic45, i)
    
## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
full_pseudo_rf45 = Model(method='rf', dataset=full_pseudo_dataset45)
full_pseudo_rf45.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_pseudo_rf45.predict_period):
    adviz_X_Y_Zrmse(full_pseudo_rf45, i)

#########################################
#### fully setting full validation

## full full validation dataset
full_full_dataset45 = Dataset(setting='full', validation='full', predict_period=30, valid_size=12, days_before=45)
full_full_dataset45.build()

## ar1
full_full_ar145 = Model(method='ar1', dataset=full_full_dataset45)
full_full_ar145.execute()


## lin reg
full_full_lin_reg45 = Model(method='lin_reg', dataset=full_full_dataset45)
full_full_lin_reg45.execute()

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
full_full_lasso45 = Model(method='lasso', dataset=full_full_dataset45)
full_full_lasso45.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_full_lasso45)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
full_full_ridge45 = Model(method='ridge', dataset=full_full_dataset45)
full_full_ridge45.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_full_ridge45)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
full_full_elastic45 = Model(method='elastic', dataset=full_full_dataset45)
full_full_elastic45.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_full_elastic45.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(full_full_elastic45, i)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
full_full_rf45 = Model(method='rf', dataset=full_full_dataset45)
full_full_rf45.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_full_rf45.predict_period):
    adviz_X_Y_Zrmse(full_full_rf45, i)

###############################################################################################
# 75 days before nowcasting
#########################################
#### Pseudo setting
## pseudo dataset
pseudo_dataset75 = Dataset(setting='pseudo', predict_period=30, valid_size=12, days_before=75)
pseudo_dataset75.build()

## ar1
pseudo_ar175 = Model(method='ar1', dataset=pseudo_dataset75)
pseudo_ar175.execute()

## lin reg
pseudo_lin_reg75 = Model(method='lin_reg', dataset=pseudo_dataset75)
pseudo_lin_reg75.execute()

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
pseudo_lasso75 = Model(method='lasso', dataset=pseudo_dataset75)
pseudo_lasso75.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(pseudo_lasso75)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
pseudo_ridge75 = Model(method='ridge', dataset=pseudo_dataset75)
pseudo_ridge75.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(pseudo_ridge75)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
pseudo_elastic75 = Model(method='elastic', dataset=pseudo_dataset75)
pseudo_elastic75.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(pseudo_elastic75.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(pseudo_elastic75, i)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
pseudo_rf75 = Model(method='rf', dataset=pseudo_dataset75)
pseudo_rf75.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(pseudo_rf75.predict_period):
    adviz_X_Y_Zrmse(pseudo_rf75, i)

#########################################
#### fully setting pseudo validation

## full pseudo validation dataset
full_pseudo_dataset75 = Dataset(setting='full', validation='pseudo', predict_period=30, valid_size=12, days_before=75)
full_pseudo_dataset75.build()

## ar1
full_pseudo_ar175 = Model(method='ar1', dataset=full_pseudo_dataset75)
full_pseudo_ar175.execute()

## lin reg
full_pseudo_lin_reg75 = Model(method='lin_reg', dataset=full_pseudo_dataset75)
full_pseudo_lin_reg75.execute()

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
full_pseudo_lasso75 = Model(method='lasso', dataset=full_pseudo_dataset75)
full_pseudo_lasso75.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_pseudo_lasso75)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
full_pseudo_ridge75 = Model(method='ridge', dataset=full_pseudo_dataset75)
full_pseudo_ridge75.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_pseudo_ridge75)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
full_pseudo_elastic75 = Model(method='elastic', dataset=full_pseudo_dataset75)
full_pseudo_elastic75.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_pseudo_elastic75.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(full_pseudo_elastic75, i)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
full_pseudo_rf75 = Model(method='rf', dataset=full_pseudo_dataset75)
full_pseudo_rf75.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_pseudo_rf75.predict_period):
    adviz_X_Y_Zrmse(full_pseudo_rf75, i)

#########################################
#### fully setting full validation

## full full validation dataset
full_full_dataset75 = Dataset(setting='full', validation='full', predict_period=30, valid_size=12, days_before=75)
full_full_dataset75.build()

## ar1
full_full_ar175 = Model(method='ar1', dataset=full_full_dataset75)
full_full_ar175.execute()

## lin reg
full_full_lin_reg75 = Model(method='lin_reg', dataset=full_full_dataset75)
full_full_lin_reg75.execute()

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
full_full_lasso75 = Model(method='lasso', dataset=full_full_dataset75)
full_full_lasso75.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_full_lasso75)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
full_full_ridge75 = Model(method='ridge', dataset=full_full_dataset75)
full_full_ridge75.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_full_ridge75)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
full_full_elastic75 = Model(method='elastic', dataset=full_full_dataset75)
full_full_elastic75.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_full_elastic75.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(full_full_elastic75, i)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
full_full_rf75 = Model(method='rf', dataset=full_full_dataset75)
full_full_rf75.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_full_rf75.predict_period):
    adviz_X_Y_Zrmse(full_full_rf75, i)