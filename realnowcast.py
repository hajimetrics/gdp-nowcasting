import pickle
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import time
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.simplefilter("ignore")

# read pkl file
f = open('master_data_xarray.pkl','rb')
master_data = pickle.load(f)
f.close

# 使う特徴量の名前
feature_names = ['GDP', 'iip_%3_1', 'cpi_%3_1',
       'exp_%3_1', 'inp_%3_1', 'rtv_%3_1', 'emp_%3_1', 'iip_%3_2',
       'cpi_%3_2', 'exp_%3_2', 'inp_%3_2', 'rtv_%3_2', 'emp_%3_2',
       'iip_%3_0', 'cpi_%3_0', 'exp_%3_0', 'inp_%3_0', 'rtv_%3_0', 'emp_%3_0']

# データを生成する Data クラス
class Dataset(object):
    def __init__(self, data=master_data, setting=None, validation=None, predict_period=30, valid_size=12, days_before=15): # setting, validationには'pseudo'か'full'. setting='pseudo'のときvalidationはNoneのままでいい
        self.data = data
        self.setting = setting
        self.validation = validation
        self.predict_period = predict_period # predict_period == validation_period == predict_period. increasing windowでの
        self.valid_size = valid_size # ケツだけvalidationのdecreasing windowの回数
        self.feature_name = self.data.coords['columns'].values # 特徴量名
        self.vintage = '2019/9/1' # データのvintage. pseudoなら'2019/9/1'. real_timeなら['1999/2/1' ~ '2019/9/1']?
        self.days_before = days_before # 何日前予測か。15日前, 45日前, 75日前のいずれか。type  int
        if self.days_before == 15:
            self.X_names = ['iip_%3_1', 'cpi_%3_1', 'exp_%3_1', 'inp_%3_1', 'rtv_%3_1', 'emp_%3_1', 
                            'iip_%3_2', 'cpi_%3_2', 'exp_%3_2', 'inp_%3_2', 'rtv_%3_2', 'emp_%3_2',
                            'iip_%3_0', 'cpi_%3_0', 'exp_%3_0', 'inp_%3_0', 'rtv_%3_0', 'emp_%3_0']# 15日前予測の変数名リスト。
        elif self.days_before == 45:
            self.X_names = ['iip_%3_1', 'cpi_%3_1', 'exp_%3_1', 'inp_%3_1', 'rtv_%3_1', 'emp_%3_1', 
                            'iip_%3_2', 'cpi_%3_2', 'exp_%3_2', 'inp_%3_2', 'rtv_%3_2', 'emp_%3_2']# 45日前予測の変数名リスト。
        elif self.days_before == 75:
            self.X_names = ['iip_%3_1', 'cpi_%3_1', 'exp_%3_1', 'inp_%3_1', 'rtv_%3_1', 'emp_%3_1']# 75日前予測の変数名リスト。  
        else:
            print(self.days_before, ' days before nowcast is not acceptable. Please select from [15, 45, 75]')
        # _gen_increase_valid_data()で定義
        self.X_valid = None # list shape = (12, 30) (valid_size * predict_period) 要素は 1次元のそれぞれの特徴量の値 (ベクトルの 計30のxr.DataArray) [[xr.DataArray]]
        self.y_valid = None # list shape = (12, 30) (valid_size * predict_period) 要素は 0次元のGDPの値 ( スカラー()の 計30のxr.DataArray) [[xr.DataArray]]
        self.X_train = None # list shape = (12, 30) (valid_size * predict_period) 要素は それぞれ残された期間の19個の特徴量 (decrese=1は (97, 19), (96, 19), ..., (69, 19), (68, 19) 計30の xr.DataArray, decrese=3 は (95, 19)~(66, 19)) [[xr.DataArray]]
        self.y_train = None # list shape = (12, 30) (valid_size * predict_period) 要素は それぞれ残された期間の1次元のGDP (decrese=1は (97,), (96,), ..., (69,), (68,) 計30の xr.DataArray) [[xr.DataArray]]
        
        # _gen_increase_test_data()で定義
        self.X_test = None # list shape = (30,) 要素は 1次元(19,)xr.DataArray [xr.DataArray]
        self.y_test = None # list shape = (30,) 要素は 0次元()xr.DataArray [xr.DataArray]
        self.X_final_train = None # list shape = (30,) 要素は それぞれ残された期間の19個の特徴量 ( (98, 19), (97, 19), ..., (70, 19), (69, 19) 計30の xr.DataArray) [xr.DataArray]
        self.y_final_train = None # list shape = (30,) 要素は 要素は ( (98,), (97,), ..., (70,), (69,) 計30の xr.DataArray) [[xr.DataArray]]
        
        # full_gen_increase_valid_data()で定義
        self.vintage_list = None # vintageの名前のリスト
        self.preliminary_vintage = None # 速報値が発表されたvintageの名前のリスト
        
        # full_fillnan()で定義
        self.firstvalue_array = None
    
    def get_available_data(self, start_period='1994-Q1'): # 使えるデータのみ抽出(1994-Q1以降) start_periodはstr
        self.data['2019/9/1'].sel(columns='rtv_%3_0')[-1] = 101.79 # 最後のrtvの欠損値の補完(####一時的に####)
        gdp_index = self.data['2019/9/1'][(self.data['2019/9/1'].sel(columns='target_gdp_time') == '1994-Q1').values == True].index.values[0] # GDPが存在する行のindex
        self.data = self.data.isel(index=slice(gdp_index, self.data.dims['index'])) # 3Ddataのスライス
        self.data.coords['index'] = np.linspace(0, self.data.coords['index'].shape[0]-1, self.data.coords['index'].shape[0], dtype='int64') # coords['index']indexを振り直し
        return None
    
    def pseudo_extract_vintage(self): # ここで3次元データから1シート抽出 xr.Dataset() --> xr.DataArray
        self.data = self.data[self.vintage]
        return None
    
    def pseudo_log_diff(self): # 対数差分変換 pseudo用 extract_vintage()後に使う
        logdiff = np.log(self.data.loc[{'columns': feature_names}].astype('float64')) - np.log(self.data.loc[{'columns': feature_names}].shift(index=1).astype('float64'))
        dlabels = self.data.loc[{'columns': ['target_gdp_time','TIME_%3_1','TIME_%3_2','TIME_%3_0']}]
        self.data = xr.concat([dlabels, logdiff], dim='columns')
        self.data = self.data.loc[{'index': self.data.coords['index'][1:].values}] #対数差分による欠損値の行の削除
        self.data.coords['index'] = np.linspace(0, self.data.coords['index'].shape[0]-1, self.data.coords['index'].shape[0], dtype='int64') # coords['index']indexを振り直し
        return None

    def pseudo_add_lag(self, feature='GDP', lag=1): # final vintage のみにラグ追加
        col_name = feature + 't-' + str(lag)
        newcol = self.data.sel(columns=feature).shift(index=lag) # 新しくラグの列
        newcol.coords['columns'] = col_name # 新しい列のcolumnsの名前を変更
        self.data = xr.concat([self.data, newcol], dim='columns')# 結合
        self.data = self.data.loc[{'index': self.data.coords['index'][1:].values}]#対数差分による欠損値の行の削除
        self.data.coords['index'] = np.linspace(0, self.data.coords['index'].shape[0]-1, self.data.coords['index'].shape[0], dtype='int64') # coords['index']indexを振り直し
        self.feature_name = self.data.coords['columns'].values # self,feature_name更新
        self.data.index.values = self.data.loc[{'columns': 'target_gdp_time'}].values # indexを数字からtarget_gdp_timeのdatetimeにする
        self.X_names.append(col_name) # 変数名リストにラグの名前を追加
        return None
    
    def pseudo_gen_increase_valid_data(self): # pseudo用 validation dataの生成。
        X = self.data.loc[{'columns': self.X_names}] # 数値データのみ抽出(日付とか年とかを落とす)
        y = self.data.loc[{'columns': 'GDP'}]
        X_valid2 = []
        y_valid2 = []
        X_train2 = []
        y_train2 = []    
        for j in range(self.valid_size): # self.valid_sizeの分だけ、validation dataの平行四辺形を伸ばす
            X_valid1 = []
            y_valid1 = []
            X_train1 = []
            y_train1 = []
            for i in range(self.predict_period): # 各predict_periodのtrain, valid dataの用意. 30セット
                X_valid1.append(X.loc[{'index': X.coords['index'].values[-(i + 2 + j)]}]) # 一番最後(最新)から二番目のものからケツだけvalidation dataにする(1番目はケツだけ予測のtest data) それぞれのperiodにおいてデータの長さが異なるため、no.arrayではなくlistに格納
                y_valid1.append(y.loc[{'index': y.coords['index'].values[-(i + 2 + j)]}])
                X_train1.append(X.loc[{'index': X.coords['index'].values[:-(i + 2 + j)]}])
                y_train1.append(y.loc[{'index': y.coords['index'].values[:-(i + 2 + j)]}])
            X_valid2.append(X_valid1)
            y_valid2.append(y_valid1)
            X_train2.append(X_train1)
            y_train2.append(y_train1)
        self.X_valid = X_valid2 # list shape = (12, 30) (valid_size * predict_period) 要素は 1次元のそれぞれの特徴量の値 (ベクトルの 計30のxr.DataArray) [[xr.DataArray]]
        self.y_valid = y_valid2 # list shape = (12, 30) (valid_size * predict_period) 要素は 0次元のGDPの値 ( スカラー()の 計30のxr.DataArray) [[xr.DataArray]]
        self.X_train = X_train2 # list shape = (12, 30) (valid_size * predict_period) 要素は それぞれ残された期間の19個の特徴量 (decrese=1は (97, 19), (96, 19), ..., (69, 19), (68, 19) 計30の xr.DataArray, decrese=3 は (95, 19)~(66, 19)) [[xr.DataArray]]
        self.y_train = y_train2 # list shape = (12, 30) (valid_size * predict_period) 要素は それぞれ残された期間の1次元のGDP (decrese=1は (97,), (96,), ..., (69,), (68,) 計30の xr.DataArray) [[xr.DataArray]]
        return None
    
    def pseudo_gen_increase_test_data(self):
        X = self.data.loc[{'columns': self.X_names}] # 数値データのみ抽出(日付とか年とかを落とす)
        y = self.data.loc[{'columns': 'GDP'}]
        X_test = []
        y_test = []
        X_final_train = []
        y_final_train = []
        for i in range(self.predict_period): # 各predict_periodのfinal_train, test dataの用意. 30セット
            X_test.append(X.loc[{'index': X.coords['index'].values[-(i + 1)]}])
            y_test.append(y.loc[{'index': y.coords['index'].values[-(i + 1)]}])
            X_final_train.append(X.loc[{'index': X.coords['index'].values[:-(i + 1)]}])
            y_final_train.append(y.loc[{'index': y.coords['index'].values[:-(i + 1)]}])
        self.X_test = X_test # list shape = (30,) 要素は 1次元(19,)xr.DataArray [xr.DataArray]
        self.y_test = y_test # list shape = (30,) 要素は 0次元()xr.DataArray [xr.DataArray]
        self.X_final_train = X_final_train # list shape = (30,) 要素は それぞれ残された期間の19個の特徴量 ( (98, 19), (97, 19), ..., (70, 19), (69, 19) 計30の xr.DataArray) [xr.DataArray]
        self.y_final_train = y_final_train # list shape = (30,) 要素は 要素は ( (98,), (97,), ..., (70,), (69,) 計30の xr.DataArray) [[xr.DataArray]]
        return None
    
    def full_log_diff(self):
        log_t = np.log(self.data.loc[{"columns": feature_names}].astype('float64')) #変数すべて対数変換
        log_t_lag1 = np.log(self.data.loc[{"columns":feature_names}].shift(index=1).astype('float64')) ##各vintageで変数すべてのt－1の対数変換
        logdiff = log_t -log_t_lag1 # 対数差分
        dlabels = self.data.loc[{'columns': ['target_gdp_time','TIME_%3_1','TIME_%3_2','TIME_%3_0']}] #ラベル取り出し
        self.data = xr.concat([dlabels, logdiff], dim='columns') #結合
        self.data = self.data.loc[{'index': self.data.coords['index'][1:].values}] #対数差分で初期値がnanになるので削除
        self.data.coords['index'] = np.linspace(0, self.data.coords['index'].shape[0]-1, self.data.coords['index'].shape[0], dtype='int64') #index振り直し
        return None
        
    def full_add_lag(self, feature="GDP", lag=1):
        col_name = feature + 't-' + str(lag)
        newcol = self.data.sel(columns=feature).shift(index=lag) # 新しくラグの列
        newcol.coords['columns'] = col_name # 新しい列のcolumnsの名前を変更
        self.data = xr.concat([self.data, newcol], dim='columns') # 結合
        self.data = self.data.loc[{'index': self.data.coords['index'][1:].values}] #t-1が存在しない初期値の行の削除
        self.data.coords['index'] = np.linspace(0, self.data.coords['index'].shape[0]-1, self.data.coords['index'].shape[0], dtype='int64') # coords['index']indexを振り直し
        self.feature_name = self.data.coords['columns'].values # self,feature_name更新 
        self.data.index.values = self.data['1999/2/1'].loc[{'columns': 'target_gdp_time'}].values # indexを数字からtarget_gdp_timeのdatetimeにする
        self.X_names.append(col_name) # 変数名リストにラグの名前を追加
        return None

    def full_gen_increase_valid_data(self): #full用 validation dataの生成。validation = 'pseudo' or 'full'
        self.vintage_list = list(self.data.data_vars.keys()) # vintageの名前のリスト
        X = self.data.loc[{'columns': self.X_names}] # xarray.Datasetで、各vintageについて数値データの特徴量のみ抽出(日付とか年とかを落とす)
        y = self.data.loc[{'columns': 'GDP'}] # xarray.Datasetで、各vintageについて目的変数であるGDPのみ抽出
        # 速報値が発表されたvintageを、欠損値数の違いで特定、抽出
        l = []
        for i in range(len(self.vintage_list)):
            if np.isnan(y[self.vintage_list[i - 1]].values.astype('float64')).sum() - np.isnan(y[self.vintage_list[i]].values.astype('float64')).sum() == 1: # 前期と比べて欠損値が1つ少なければ、その期にGDP速報値が発表されていることになる。
                l.append(self.vintage_list[i])
        l = np.array(l)
        target = list(self.data.loc[{'columns': 'target_gdp_time'}]['1999/2/1'].values) # 全対象GDPの名前リスト
        self.preliminary_vintage = np.array([target[-len(l):], l]).T # GDPの速報値が発表されたvintageの名前array 長さは83しかないが、最初のほうはそもそも予測の対象にならないので、速報値vintageがなくても大丈夫
        self.preliminary_vintage[np.where(self.preliminary_vintage[:, 0] == '2013-Q1'), 1] = '2013/6/1' # 2013-Q1の速報値は5/16に発表。
        self.preliminary_vintage_test_range = self.preliminary_vintage[-self.predict_period:] # testデータに使うvintageの名前array
        vr = []
        for i in range(self.valid_size):
            vr.append(self.preliminary_vintage[-self.predict_period-(i + 1): -(i + 1)])
        self.preliminary_vintage_valid_range = np.array(vr) # validデータに使うvintageの名前array (valid_size分) 各predict_periodに対し、validationのvintageはvalid_sizeごとに事なる。
        
        if self.validation == 'pseudo': # pseudo validation
            X_valid2 = []
            y_valid2 = []
            X_train2 = []
            y_train2 = []
            for j in range(self.valid_size):
                X_valid1 = []
                y_valid1 = []
                X_train1 = []
                y_train1 = []
                for i in range(self.predict_period):
                    y_test_vintage = self.preliminary_vintage_test_range[:, 1][i] # y_test_vintageは速報値発表vintage
                    if self.days_before == 15: # X_valid, y_valid, X_train, y_train, (X_test)のvintageはすべてy_test_vintageの1ヶ月前(15days)
                        pseudo_valid_vintage = datetime.strptime(y_test_vintage, '%Y/%m/%d') - relativedelta(months=1)
                    elif self.days_before == 45: # X_valid, y_valid, X_train, y_train, (X_test)のvintageはすべてy_test_vintageの2ヶ月前(45days)
                        pseudo_valid_vintage = datetime.strptime(y_test_vintage, '%Y/%m/%d') - relativedelta(months=2)
                    elif self.days_before == 75: # X_valid, y_valid, X_train, y_train, (X_test)のvintageはすべてy_test_vintageの3ヶ月前(75days)
                        pseudo_valid_vintage = datetime.strptime(y_test_vintage, '%Y/%m/%d') - relativedelta(months=3)
                    else:
                        print(self.days_before, ' days before nowcast is not acceptable.')
                        break
                    pseudo_valid_vintage = str(pseudo_valid_vintage.year) + '/' + str(pseudo_valid_vintage.month) + '/' + str(pseudo_valid_vintage.day)
                    valid_target = self.preliminary_vintage_valid_range[j][i][0] # valid_size j番目の、predict_period i番目の予測タイミングvintageで予測する対象GDP
                    X_valid1.append(X[pseudo_valid_vintage].loc[{'index': valid_target}]) # 正解valid_vintageの時に使える予測対象GDPに関する特徴量(ベクトル) 
                    y_valid1.append(y[pseudo_valid_vintage].loc[{'index': valid_target}]) # 正解valid_vintageの時の予測対象GDP(スカラー)
                    train_target = target[: target.index(valid_target)] # 全対象GDP target から valid_target 以降の対象GDPを除いた train_target
                    X_train1.append(X[pseudo_valid_vintage].loc[{'index': train_target}]) # train_vintageで使える予測対象GDPに関する特徴量(行列)
                    y_train1.append(y[pseudo_valid_vintage].loc[{'index': train_target}]) # train_vintageで使える予測対象GDP(ベクトル)
                X_valid2.append(X_valid1)
                y_valid2.append(y_valid1)
                X_train2.append(X_train1)
                y_train2.append(y_train1)
            self.X_valid = X_valid2 # list shape = (12, 30) (valid_size * predict_period) 要素は 1次元のそれぞれの特徴量の値 (ベクトルの 計30のxr.DataArray) [[xr.DataArray]]
            self.y_valid = y_valid2 # list shape = (12, 30) (valid_size * predict_period) 要素は 0次元のGDPの値 ( スカラー()の 計30のxr.DataArray) [[xr.DataArray]]
            self.X_train = X_train2 # list shape = (12, 30) (valid_size * predict_period) 要素は それぞれ残された期間の19個の特徴量 (decrese=1は (69, 19), (70, 19), ..., (97, 19), (98, 19) 計30の unbalanced xr.DataArray
            self.y_train = y_train2 # list shape = (12, 30) (valid_size * predict_period) 要素は それぞれ残された期間の1次元のGDP (decrese=1は (68,), (70,), ..., (97,), (98,) 計30の unbalanced xr.DataArray)
            return None
        elif self.validation == 'full': # full validation
            X_valid2 = []
            y_valid2 = []
            X_train2 = []
            y_train2 = []
            for j in range(self.valid_size):
                X_valid1 = []
                y_valid1 = []
                X_train1 = []
                y_train1 = []
                for i in range(self.predict_period):
                    y_valid_vintage = self.preliminary_vintage_valid_range[j][i][1] # valid_size j番目の、predict_period i番目 の y_validのgdpの速報値。これがy_valid_vintageのvintage
                    if self.days_before == 15: # X_valid, X_train, y_trainのvintageは y_valid_vintage の1ヶ月前(15days)
                        X_valid_vintage = datetime.strptime(y_valid_vintage, '%Y/%m/%d') - relativedelta(months=1)
                    elif self.days_before == 45: # X_valid, X_train, y_trainのvintageは y_valid_vintage の2ヶ月前(45days)
                        X_valid_vintage = datetime.strptime(y_valid_vintage, '%Y/%m/%d') - relativedelta(months=2)
                    elif self.days_before == 75: # X_valid, X_train, y_trainのvintageは y_valid_vintage の3ヶ月前(75days)
                        X_valid_vintage = datetime.strptime(y_valid_vintage, '%Y/%m/%d') - relativedelta(months=3)
                    else:
                        print(self.days_before, ' //days before nowcast is not acceptable.')
                        break                        
                    X_valid_vintage = str(X_valid_vintage.year) + '/' + str(X_valid_vintage.month) + '/' + str(X_valid_vintage.day)
                    valid_target = self.preliminary_vintage_valid_range[j][i][0] # valid_size j番目の、predict_period i番目の予測タイミングvintageで予測する対象GDP
                    y_valid1.append(y[y_valid_vintage].loc[{'index': valid_target}]) # 正解valid_vintageの時の予測対象GDP(スカラー)
                    X_valid1.append(X[X_valid_vintage].loc[{'index': valid_target}]) # 正解valid_vintageの時に使える予測対象GDPに関する特徴量(ベクトル) 
                    train_target = target[: target.index(valid_target)] # 全対象GDP target から valid_target 以降の対象GDPを除いた train_target
                    X_train1.append(X[X_valid_vintage].loc[{'index': train_target}]) # train_vintageで使える予測対象GDPに関する特徴量(行列)
                    y_train1.append(y[X_valid_vintage].loc[{'index': train_target}]) # train_vintageで使える予測対象GDP(ベクトル)
                X_valid2.append(X_valid1)
                y_valid2.append(y_valid1)
                X_train2.append(X_train1)
                y_train2.append(y_train1)
            self.X_valid = X_valid2 # list shape = (12, 30) (valid_size * predict_period) 要素は 1次元のそれぞれの特徴量の値 (ベクトルの 計30のxr.DataArray) [[xr.DataArray]]
            self.y_valid = y_valid2 # list shape = (12, 30) (valid_size * predict_period) 要素は 0次元のGDPの値 ( スカラー()の 計30のxr.DataArray) [[xr.DataArray]]
            self.X_train = X_train2 # list shape = (12, 30) (valid_size * predict_period) 要素は それぞれ残された期間の19個の特徴量 (decrese=1は (69, 19), (70, 19), ..., (97, 19), (98, 19) 計30の unbalanced xr.DataArray
            self.y_train = y_train2 # list shape = (12, 30) (valid_size * predict_period) 要素は それぞれ残された期間の1次元のGDP (decrese=1は (68,), (70,), ..., (97,), (98,) 計30の unbalanced xr.DataArray)
            return None
        else:
            print('No such setting.')
    
    def full_gen_increase_test_data(self):
        X = self.data.loc[{'columns': self.X_names}] # xarray.Datasetで、各vintageについて数値データの特徴量のみ抽出(日付とか年とかを落とす)
        y = self.data.loc[{'columns': 'GDP'}] # xarray.Datasetで、各vintageについて目的変数であるGDPのみ抽出
        target = list(self.data.loc[{'columns': 'target_gdp_time'}]['1999/2/1'].values) # 全対象GDPの名前リスト
        X_test = []
        y_test = []
        X_final_train = []
        y_final_train = []
        for i in range(self.predict_period): # 各predict_periodのfinal_train, test dataの用意. 30セット
            y_test_vintage = self.preliminary_vintage_test_range[:, 1][i] # y_test_vintageは速報値発表vintage
            test_target = self.preliminary_vintage_test_range[:, 0][i]
            y_test.append(y[y_test_vintage].loc[{'index': test_target}])
            if self.days_before == 15: # X_test, X_final_train, y_final_trainのvintageはy_test_vintageの1ヶ月前(15days)
                X_test_vintage = datetime.strptime(y_test_vintage, '%Y/%m/%d') - relativedelta(months=1)
            elif self.days_before == 45: # X_test, X_final_train, y_final_trainのvintageはy_test_vintageの2ヶ月前(45days)
                X_test_vintage = datetime.strptime(y_test_vintage, '%Y/%m/%d') - relativedelta(months=2)
            elif self.days_before == 75: # X_test, X_final_train, y_final_trainのvintageはy_test_vintageの3ヶ月前(75days)
                X_test_vintage = datetime.strptime(y_test_vintage, '%Y/%m/%d') - relativedelta(months=3)
            else:
                print(self.days_before, ' days before nowcast is not acceptable.')
                break
            X_test_vintage = str(X_test_vintage.year) + '/' + str(X_test_vintage.month) + '/' + str(X_test_vintage.day)
            final_train_target = target[: target.index(test_target)]
            X_test.append(X[X_test_vintage].loc[{'index': test_target}]) # X_testのvintageはfinal_trainと同じ
            X_final_train.append(X[X_test_vintage].loc[{'index': final_train_target}])
            y_final_train.append(y[X_test_vintage].loc[{'index': final_train_target}])
        self.X_test = X_test # list shape = (30,) 要素は 1次元(19,)xr.DataArray [xr.DataArray]
        self.y_test = y_test # list shape = (30,) 要素は 0次元()xr.DataArray [xr.DataArray]
        self.X_final_train = X_final_train # list shape = (30,) 要素は それぞれ残された期間の19個の特徴量 ( (98, 19), (97, 19), ..., (70, 19), (69, 19) 計30の xr.DataArray) [xr.DataArray]
        self.y_final_train = y_final_train # list shape = (30,) 要素は 要素は ( (98,), (97,), ..., (70,), (69,) 計30の xr.DataArray) [[xr.DataArray]]
        return None
 
    def full_fillnan(self): # 欠損値の補填
        # 速報値発表リストの作成
        Xs = ['iip_%3_1', 'cpi_%3_1', 'exp_%3_1', 'inp_%3_1', 'rtv_%3_1', 'emp_%3_1', 'iip_%3_2', 'cpi_%3_2', 'exp_%3_2', 'inp_%3_2', 'rtv_%3_2', 'emp_%3_2', 'iip_%3_0', 'cpi_%3_0', 'exp_%3_0', 'inp_%3_0', 'rtv_%3_0', 'emp_%3_0'] 
        vintageX = ['vin_iip_%3_1','iip_%3_1','vin_cpi_%3_1','cpi_%3_1','vin_exp_%3_1','exp_%3_1','vin_inp_%3_1','inp_%3_1', 'vin_rtv_%3_1','rtv_%3_1','vin_emp_%3_1','emp_%3_1','vin_iip_%3_2','iip_%3_2','vin_cpi_%3_2','cpi_%3_2','vin_exp_%3_2', 'exp_%3_2','vin_inp_%3_2','inp_%3_2','vin_rtv_%3_2','rtv_%3_2','vin_emp_%3_2','emp_%3_2','vin_iip_%3_0','iip_%3_0', 'vin_cpi_%3_0','cpi_%3_0','vin_exp_%3_0','exp_%3_0','vin_inp_%3_0','inp_%3_0','vin_rtv_%3_0','rtv_%3_0','vin_emp_%3_0', 'emp_%3_0'] 
        X = self.data.loc[{'columns': Xs}]
        vintage_list = self.vintage_list
        all_list = []
        for variable in Xs:
            firstva_l = []
            firstvin_l = []
            for i in range(len(X.index)):
                arr = X.isel(index=i).loc[{"columns":variable}].to_array().values.astype("float64")
                arrnan = np.isnan(arr)
                for j in range(len(arr)):
                    if arrnan[j] == True:
                        pass
                    else:
                        firstva_l.append(arr[j])
                        firstvin_l.append(vintage_list[j])
                        break
            all_list.append(firstvin_l)
            all_list.append(firstva_l)
        firstval_array = xr.DataArray(all_list,dims=["columns","target"], coords=[vintageX,X.index.astype("str")]).T
        self.firstvalue_array = firstval_array
        # X_train, X_validの補填
        for l in range(self.valid_size):
            for k in range(self.predict_period):
                # X_trainの補填
                da = self.X_train[l][k]
                n = da.columns.values * np.isnan(da.values.astype('float64'))
                for j in range(n.shape[0]): # index loop
                    for i in range(n.shape[1]): # columns loop except
                        if n[j, i] == '':
                            pass
                        else:
                            tgt = da.index.values[j]
                            var = da.columns.values[i]
                            self.X_train[l][k].loc[{'index': tgt, 'columns': var}] = self.firstvalue_array.loc[{'columns': var, 'target': tgt}].values.astype('float64')
                # X_validの補填
                da = self.X_valid[l][k]
                n = da.columns.values * np.isnan(da.values.astype('float64'))
                for i in range(n.shape[0]): # columns loop except
                    if n[i] == '':
                        pass
                    else:
                        tgt = da.index.values
                        var = da.columns.values[i]
                        self.X_valid[l][k].loc[{'columns': var}] = self.firstvalue_array.loc[{'columns': var, 'target': tgt}].values.astype('float64')
        # X_final_train, X_testの補填
        for k in range(self.predict_period):
            # X_final_trainの補填
            da = self.X_final_train[k]
            n = da.columns.values * np.isnan(da.values.astype('float64'))
            for j in range(n.shape[0]): # index loop
                for i in range(n.shape[1]): # columns loop
                    if n[j, i] == '':
                        pass
                    else:
                        tgt = da.index.values[j]
                        var = da.columns.values[i]
                        self.X_final_train[k].loc[{'index': tgt, 'columns': var}] = self.firstvalue_array.loc[{'columns': var, 'target': tgt}].values.astype('float64')
            # X_testの補填
            da = self.X_test[k]
            n = da.columns.values * np.isnan(da.values.astype('float64'))
            for i in range(n.shape[0]): # columns loop
                if n[i] == '':
                    pass
                else:
                    tgt = da.index.values
                    var = da.columns.values[i]
                    self.X_test[k].loc[{'columns': var}] = self.firstvalue_array.loc[{'columns': var, 'target': tgt}].values.astype('float64')
        return None
    
    def build(self): # settingごとに上のmethodをまとめて実行。setting='full'の場合、validationを'pseudo'か'full'を選ぶ
        self.get_available_data()
        if self.setting == 'pseudo':
            self.pseudo_extract_vintage()
            self.pseudo_log_diff()
            self.pseudo_add_lag()
            self.pseudo_gen_increase_valid_data()
            self.pseudo_gen_increase_test_data()
            return None
        elif self.setting == 'full':
            self.full_log_diff()
            self.full_add_lag()
            self.full_gen_increase_valid_data()
            self.full_gen_increase_test_data()
            self.full_fillnan()
            return None
        else:
            print("No such setting")
            return None
    
# 全てのモデルを生成できる Model クラス
class Model(object): 
    def __init__(self, method, dataset):
        self.method = method # モデルの種類 ['ar1', 'lin_reg', 'lasso', 'ridge', 'elastic', 'dt', 'rf'] のどれか
        self.dataset = dataset # type Dataset()
        self.predict_period = self.dataset.predict_period
        self.valid_size = self.dataset.valid_size
        
        # set_increase_valid_model()で定義
        self.hyparam_list1 = None
        self.hyparam_list2 = None
        self.hyparam_array = None
        self.valid_models = None # validation用のモデルリスト (sklearnのオブジェクト) ウィンドウ動かさないならながさ1のリスト
        
        # increase_validation()で定義
        self.y_valid_hat = None # validationの予測値のndarray shape=(valid_size, predict_period, hyparam_list)
        self.valid_error = None # validationの予測誤差のndarray shape=(hyparam_list, valid_size, predict_period)
        self.valid_mse = None # validationの予測の mean squared errorのndarray shape=(hyparam_list, predict_period)
        self.valid_rmse = None # validation の root mse
        
        # increase_validation()で定義
        self.best_tuned_model = None # validation後のベストモデル一個のみ
        self.best_hyparam = None # validation後にわかる一番いいハイパーパラメータ
        self.best_y_hat = None
        self.bset_error = None
        self.best_mse = None
        self.best_rmse = None
        self.best_hyparam_index = None
        
        # set_increase_test_model()で定義
        self.test_models = None # 最終fit用のモデルリスト (sklearnのオブジェクト)
        
        # increase_test()で定義
        self.y_test_hat = None # testの予測値のarray
        self.test_error = None # testの予測誤差のarray
        self.test_mse = None # testの予測の mean squared error
        self.test_rmse = None # test の root mse
        
        
    def set_increase_valid_model(self, hyparam_list1=None, hyparam_list2=None): # predict_period, valid_sizeは、Dataset.pseudo_gen_increase_valid_data()にそろえる! predict_period(==valid_period) * valid_size * hyparam_list (30 * 3 * 6) のモデルの3次元np.arrayを生成 。'rf'だけ、tuning paramの質が違うので注意
        # 'ar1'と'lin_reg'は validation しない
        self.hyparam_list1 = hyparam_list1
        self.hyparam_list2 = hyparam_list2 # hyparamsが二つあるときのみ入力. なければNone
        
        if self.method == 'lasso': # hyparam は alpha
            ml_1d = [] # self.hyparam_listそれぞれに対応する未fitモデルのnp.arrayの生成. shape = (6,)
            for hyparam1_ in self.hyparam_list1:
                ml_1d.append(Lasso(hyparam1_))
            ml_1d = np.array(ml_1d)
            ml_2d = np.array([ml_1d] * self.predict_period) # predict_periodに対し、ml_1d から次元を追加して未fitモデルのnp.arrayを拡張. shape = (30, 6)
            ml_3d = np.array([ml_2d] * self.valid_size) # valid_sizeに対し、ml_2d から次元を追加して未fitモデルのnp.arrayを拡張. shape = (3, 30, 6)
            self.valid_models = ml_3d
            #self.valid_models = np.array([model_list] * valid_window) # validation windowごとにfitさせるため、model_listをvalid_window分用意. modelのmatrixを生成。np.array.shape=(30 * 6)
            return None
        
        elif self.method == 'ridge':
            ml_1d = [] # self.hyparam_listそれぞれに対応する未fitモデルのnp.arrayの生成. shape = (6,)
            for hyparam1_ in self.hyparam_list1:
                ml_1d.append(Ridge(hyparam1_))
            ml_1d = np.array(ml_1d)
            ml_2d = np.array([ml_1d] * self.predict_period) # predict_periodに対し、ml_1d から次元を追加して未fitモデルのnp.arrayを拡張. shape = (30, 6)
            ml_3d = np.array([ml_2d] * self.valid_size) # valid_sizeに対し、ml_2d から次元を追加して未fitモデルのnp.arrayを拡張. shape = (3, 30, 6)
            self.valid_models = ml_3d
            return None
        
        elif self.method == 'elastic':
            hl = []
            ml_1d = []
            for hyparam2_ in self.hyparam_list2:
                ml_0d = [] 
                for hyparam1_ in self.hyparam_list1:
                    hl.append([hyparam1_, hyparam2_])
                    ml_0d.append(ElasticNet(alpha=hyparam1_, l1_ratio=hyparam2_)) # hp1: alpha, hp2: l1_ratio
                ml_1d.append(ml_0d)
            self.hyparam_array = np.array(hl) # 試したhyparamsの組のarray shape = (36, 2)
            ml_1d = np.array(ml_1d) # self.hyparam_list1, self.hyparam_list2それぞれに対応する未fitモデルのnp.arrayの生成. shape = (6, 6)
            ml_2d = np.array([ml_1d] * self.predict_period) # predict_periodに対し、ml_1d から次元を追加して未fitモデルのnp.arrayを拡張. shape = (30, 6, 6)
            ml_3d = np.array([ml_2d] * self.valid_size) # valid_sizeに対し、ml_2d から次元を追加して未fitモデルのnp.arrayを拡張. shape = (3, 30, 6, 6)
            self.valid_models = ml_3d.reshape(self.valid_size, self.predict_period, -1) # 3次元に変形(後のfitを, hyparamが1つの時と統一するため) shape = (3, 30, 36)
            return None
        
        elif self.method == 'rf':
            hl = []
            ml_1d = []
            for hyparam2_ in self.hyparam_list2:
                ml_0d = [] 
                for hyparam1_ in self.hyparam_list1:
                    hl.append([hyparam1_, hyparam2_])
                    ml_0d.append(RandomForestRegressor(n_estimators=int(hyparam1_), max_depth=int(hyparam2_), max_features=0.33, random_state=0)) # hp1: n_estimator, hp2: max_depth
                ml_1d.append(ml_0d)
            self.hyparam_array = np.array(hl) # 試したhyparamsの組のarray shape = (36, 2)
            ml_1d = np.array(ml_1d) # self.hyparam_list1, self.hyparam_list2それぞれに対応する未fitモデルのnp.arrayの生成. shape = (6, 6)
            ml_2d = np.array([ml_1d] * self.predict_period) # predict_periodに対し、ml_1d から次元を追加して未fitモデルのnp.arrayを拡張. shape = (30, 6, 6)
            ml_3d = np.array([ml_2d] * self.valid_size) # valid_sizeに対し、ml_2d から次元を追加して未fitモデルのnp.arrayを拡張. shape = (3, 30, 6, 6)
            self.valid_models = ml_3d.reshape(self.valid_size, self.predict_period, -1) # 3次元に変形(後のfitを, hyparamが1つの時と統一するため) shape = (3, 30, 36)
            return None
        
        else:
            print('!Error: no such method!')
            #
            return None
        
    def increase_validation(self): #self.valid_modelsを用いて実際にvalidation
        # 実際にfit
        # 階層を潜ってモデルをフィットしてpredictする
        yvh3 = []
        for decrease_ in range(self.valid_models.shape[0]): # これらをdecrease分ずらして繰り返す
            yvh2 = []
            for period_ in range(self.valid_models.shape[1]): # あるdecreaseにおいて、各予測periodに対応するtrainデータを生成
                X_train_ = self.dataset.X_train[decrease_][period_].values.astype('float64')
                y_train_ = self.dataset.y_train[decrease_][period_].values.astype('float64')
                yvh1 = []
                for i in range(self.valid_models.shape[2]): # 各hyparam_listの要素ごとに作成されたモデルをループ
                    self.valid_models[decrease_][period_][i].fit(X_train_, y_train_)
                    X_valid_ = self.dataset.X_valid[decrease_][period_].values.reshape(1, -1)
                    yvh1.append(self.valid_models[decrease_][period_][i].predict(X_valid_))
                yvh2.append(yvh1)
            yvh3.append(yvh2)
        self.y_valid_hat = np.array(yvh3).reshape(self.valid_models.shape) # np.arrayにしておくと、test_error, mse, rmseの計算が楽

        # valid_error の計算
        ve = []
        for i in range(self.valid_models.shape[2]):
            ve.append(np.array(self.dataset.y_valid)[:, :] - self.y_valid_hat[:, :, i])
        self.valid_error = np.array(ve) # shape(num_valid_models, valid_size, predict_period)
        self.valid_mse = np.square(self.valid_error).mean(axis=1) # valid_sizeで平均
        self.valid_rmse = self.valid_mse ** (1 / 2)
        self.best_mse = np.amin(self.valid_mse, axis=0) # 各predict_periodそれぞれに対する最小mse. ndarray shape=(predict_period,)
        self.best_rmse = np.amin(self.valid_rmse, axis=0)
        self.best_hyparam_index = np.argmin(self.valid_mse, axis=0) # 各prediction_periodの最小mseの、hyparam_listにおけるindex
        
        # mse最小にするhyparamの特定
        # hyparam_list2 == None のとき
        if self.hyparam_list2 == None:
            print('No 2nd hyper parameter')
            self.best_hyparam = np.array([self.hyparam_list1] * self.predict_period)[ np.linspace(0, self.predict_period-1, self.predict_period, dtype='int').reshape(-1, 1) , self.best_hyparam_index.reshape(-1, 1)] # self.best_hyparam_indexをもとに、各periodに対応するbest_hyparamをself.hyparam_listをperiod倍したarrayから抽出
            return None
        # hyparam_list2が入力されているとき
        else:
            print('2nd hyper parameter exist')
            self.best_hyparam = self.hyparam_array[self.best_hyparam_index] # self.best_hyparam_indexをもとに、各periodに対応するbest_hyparamをself.hyparam_arrayから抽出
            return None
        
    def set_increase_test_model(self): # predict_periodの数だけモデルを用意する, self.test_modelsにはモデルを格納したarrayが代入される(dataset, validationとpredict_periodをそろえる)
        if self.method == 'ar1':
            self.test_models = np.array([LinearRegression()] * self.predict_period) # predict_periodに対し、未fitモデルのnp.arrayを生成. shape = (30,)
            return None
        
        elif self.method == 'lin_reg':
            self.test_models = np.array([LinearRegression()] * self.predict_period) # ar1と同じ。入れるデータが違うだけ
            return None
        
        elif self.method == 'lasso':
            models = []
            for i in self.best_hyparam:
                models.append(Lasso(alpha=i))
            self.test_models = np.array(models) # hyparam は alpha
            return None
        
        elif self.method == 'ridge':
            models = []
            for i in self.best_hyparam:
                models.append(Ridge(alpha=i))
            self.test_models = np.array(models) # hyparam は alpha
            return None
        
        elif self.method == 'elastic':
            models = []
            for i in self.best_hyparam:
                models.append(ElasticNet(alpha=i[0], l1_ratio=i[1])) ##現状l1_ratio=0.5で固定##
            self.test_models = np.array(models) # hyparam は alpha, l1_ratio
            return None
        
        elif self.method == 'dt': # 決定木は保留
            ###
            return None
        
        elif self.method == 'rf':
            models = []
            for i in self.best_hyparam:
                models.append(RandomForestRegressor(n_estimators=int(i[0]), max_depth=int(i[1]), max_features=0.33, random_state=0)) ##現状max_depth=3で固定##
            self.test_models = np.array(models) # hyparam は n_estimators, max_depth
            return None
        
        else:
            print('!Error: no such method!')
            #
            return None
    
    def increase_test(self): # 最後の increase window の test の予測
        # fit
        # ar1の場合、X_final_trainはgdpのみ
        if self.method == 'ar1':
            yth = []
            for period_ in range(len(self.test_models)):
                X_final_train_ = self.dataset.X_final_train[period_].loc[{'columns': 'GDPt-1'}].values.astype('float64').reshape(-1, 1)
                y_final_train_ = self.dataset.y_final_train[period_].values.astype('float64')
                self.test_models[period_].fit(X_final_train_, y_final_train_)
                X_test_ = self.dataset.X_test[period_].loc[{'columns': 'GDPt-1'}].values.reshape(1, 1)
                yth.append(self.test_models[period_].predict(X_test_))

        # それ以外は、X_final_trainを全部使う
        else:
            yth = []
            for period_ in range(len(self.test_models)):
                X_final_train_ = self.dataset.X_final_train[period_].values
                y_final_train_ = self.dataset.y_final_train[period_].values
                self.test_models[period_].fit(X_final_train_, y_final_train_)
                X_test_ = self.dataset.X_test[period_].values.reshape(1, -1)
                yth.append(self.test_models[period_].predict(X_test_))

        # それぞれのattributesを定義
        self.y_test_hat = np.array(yth).reshape(-1)
        self.test_error = np.array(self.dataset.y_test) - self.y_test_hat
        self.test_mse = np.square(self.test_error).mean()
        self.test_rmse = np.sqrt(self.test_mse)
        return None
    
    def show(self):
        print("****************************************************************")
        print("* setting: ", self.dataset.setting)
        print("* validation: ", self.dataset.validation)
        print("* days before", self.dataset.days_before)
        print("* method: ", self.method)
        print("* hyparam_list: ", self.hyparam_list1)
        print("* hyparam_list2: ", self.hyparam_list2)
        print("* best_hyparam: ", self.best_hyparam)
        print("* test_mse: ", self.test_mse)
        print("* test_rmse", self.test_rmse)
        print("****************************************************************")
        
    def execute(self, hpl1=None, hpl2=None):
        if self.method == 'ar1' or self.method == 'lin_reg': # validationなし
            t1 = time.time()
            self.set_increase_test_model()
            self.increase_test()
            t2 = time.time()
            self.show()
            print('elapsed time: ', t2 - t1)
        else: # validationあり
            t1 = time.time()
            self.set_increase_valid_model(hyparam_list1=hpl1, hyparam_list2=hpl2)
            self.increase_validation()
            self.set_increase_test_model()
            self.increase_test()
            t2 = time.time()
            self.show()
            print('elapsed time: ', t2 - t1)