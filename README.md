機械学習を用いた日本のGDPナウキャスト
====
## データの発表およびナウキャストのタイミング
![スライド1](https://user-images.githubusercontent.com/59826800/72252220-40230600-3642-11ea-9721-88d61c0d7fef.PNG)

## 疑似コード
### 疑似ナウキャスト

![alg1](https://user-images.githubusercontent.com/59720853/72235602-6c289200-3616-11ea-9360-b869d0dff561.png)

### 予測時点版検証を用いた真正ナウキャスト

![alg2](https://user-images.githubusercontent.com/59826800/72265795-73748d80-3660-11ea-8aa8-ec4ca625daef.png)

### 1次速報値データ検証を用いた真正ナウキャスト

![alg3](https://user-images.githubusercontent.com/59720853/72235631-95e1b900-3616-11ea-927d-0c88de692dfd.png)



## ファイルの概要
### main.py
realnowcast.pyで定義したクラスを用いて、15日前、45日前、75日前の疑似・真正ナウキャストを実行する。realnowcast.Mode.show()でプリントされる結果が表示される。ナウキャストの前に検証(バリデーション)で用いるハイパーパラメータのリストを与える。各検証の様子をvisualize.pyで定義した関数を用いて図示する。

### realnowcast.py

#### `class Dataset(self, data="master_data_xarray.pkl", setting=None, validation=None, predict_period=30, valid_size=12, days_before=15)`
GDPナウキャストを行うために使うデータセットオブジェクト。
##### Parameters
* `data`: (type: xarray.Dataset)
用いる元データ。デフォルトでmasterdata_data_xarray.pklを読み込む。
* `setting`: (type: str)
疑似ナウキャスト用データセットの場合は`pesudo`、真正ナウキャスト用データセットの場合は`full`と入力する。
* `validation`: (type: str)  
真正ナウキャストにおける検証の戦略。予測時点版検証なら`pseudo`、1次速報値データ検証なら`full`と入力する。
* `predict_period`: (type: int) 
ナウキャスト対象期の数。
* `valid_size`: (type: int)  
検証データに用いる期の数。
* `days_before`: (type: int)  
何日前ナウキャストかを指定。`15`, `45`, `75`のいずれか。
##### Methods
* `get_available_data(self, start_period='1994-Q1')`:  
用いるデータの開始時点を指定して抽出する。
* `pseudo_extract_vintage(self)`:  
疑似ナウキャスト用。`self.vintage`を参照して1シート抽出し、`self.data`とする。
* `pseudo_log_diff(self)`:  
疑似ナウキャスト用。`self.data`を対数差分変換する。サンプルサイズが1つ減る。
* `pseudo_add_lag(self, feature='GDP', lag=1)`:  
疑似ナウキャスト用。特徴量の名前とラグを指定して新しく`self.data`に追加します。サンプルサイズが1つ減る。
* `pseudo_gen_increase_valid_data(self)`:  
疑似ナウキャスト用。予測の期間(testもvalidationも同じ長さ)、検証の回数を指定して、検証用の`self.X_train`, `self.y_train`, `self.X_valid`, `self.y_valid`を定義する。
* `pseudo_gen_increase_test_data(self)`:  
疑似ナウキャスト用。検証用データと同じ期間の長さで`self.X_final_train`, `y_final_train`, `self.X_test`, `self.y_test`を定義する。
* `full_log_diff(self)`:  
真正ナウキャスト用。`self.data`を対数差分変換する。サンプルサイズが1つ減る。
* `full_add_lag(self, feature='GDP', lag=1)`:  
真正ナウキャスト用。特徴量の名前とラグを指定して新しく`self.data`に追加します。サンプルサイズが1つ減る。
* `full_gen_increase_valid_data(self)`:  
真正ナウキャスト用。予測の期間(testもvalidationも同じ長さ)、検証の回数を指定して、検証用の`self.X_train`, `self.y_train`, `self.X_valid`, `self.y_valid`を定義する。
* `full_gen_increase_test_data(self)`:  
真正ナウキャスト用。検証用データと同じ期間の長さで`self.X_final_train`, `y_final_train`, `self.X_test`, `self.y_test`を定義する。
* `full_fillnan(self)`:  
真正ナウキャスト用。欠損値の処理を行う。  
* `build(self)`:  
パラメータで指定したsettingごとに上記のメソッドを使い分けてDatasetオブジェクトを作成する。
##### Attributes
###### <クラス作成時に定義>  
* `self.data`:  
データ本体。デフォルトで"master_data_xarray.pkl"を読み込む。
* `self.setting`:  
疑似ナウキャストか真正ナウキャストか。
* `self.validation`:  
真正ナウキャストの検証の戦略
* `self.predict_period`:  
ナウキャスト対象期の数。
* `self.valid_size`:  
検証データに用いる期の数。
* `self.feature_name`:  
説明変数名のリスト。
* `self.vintage`:  
`Dataset.extracet_vintage()`で参照する疑似ナウキャストのvintage。デフォルト`'2019/9/1'`
* `self.days_before`: 
何日前予測か。  
###### <`_gen_increase_valid_data()`で定義>  
* `self.X_valid`:  
検証データの説明変数。(検証のテスト用)
* `self.y_valid`:  
検証データの目的変数。(検証のテスト用)
* `self.X_train`:  
検証用推定データの説明変数。(検証の訓練用)
* `self.y_train`:  
検証用推定データの目的変数。(検証の訓練用)  
###### <`_gen_increase_test_data()`で定義>  
* `self.X_test`:  
テストデータの説明変数。
* `self.y_test`:  
テストデータの目的変数。
* `self.X_final_train`:  
最終的な訓練データの説明変数。
* `self.y_final_train`:  
最終的な訓練データの目的変数。  
###### <`full_gen_increase_valid_data()`で定義>  
* `self.vintage_list`:  
vintageの名前のリスト。  
* `self.preliminary_vintage`:  
速報値が発表されたvintageの名前のリスト。  
###### <`full_fillnan()`で定義>  
* `self.firstvalue_array`:  
欠損値処理のための各変数の速報値リスト。

#### `class Model(self, method, dataset)`
ナウキャストをする際のモデルオブジェクト。
##### Parameters
* `method`: (type: str)  
手法を入力。`'ar1'`, `'lin_reg'`, `'lasso'`, `'ridge'`, `'elastic'`, `'rf'`のいずれかを入力。  
* `dataset`: (type: realnowcast.Dataset)  
`realnowcast.Dataset`クラスオブジェクトを引数に渡す。  
##### Methods
* `set_increase_valid_model(self, hyparam_list1=None, hyparam_list2=None)`:  
検証用に、入力したハイパーパラメータリストを元に3次元のモデルのarrayである`self.valid_models`を定義する。
* `increase_validation(self)`:  
`set_increase_valid_model()`で用意した未fitモデル配列`self.valid_models`を実際にfitし、`self.valid_mse`を算出し、`self.best_hyparam`を求める。
* `set_increase_test_model(self)`:  
テスト用に、`increase_validation()`で求めた`self.best_hyparam`を用いて`predict_period`分の1次元のモデル配列`self.test_models`を定義する。
* `increase_test(self)`:  
`set_increase_test_model()`で用意した未fitモデル`self.test_models`を実際にfitし、`self.test_mse`を算出。
* `show(self)`:  
結果のプリント。
* `execute(self, hpl1=None, hpl2=None)`:  
上記のメソッドをまとめて実行する。
##### Attributes
###### <クラス作成時に定義>  
* `self.method`:  
モデルの手法。
* `self.dataset`:  
モデル推定に用いるデータ(`realnowcast.Dataset`オブジェクト)
* `self.predict_period`:  
ナウキャスト対象期の数。
* `self.valid_size`:  
検証データに用いる期の数。  
###### <`set_increase_valid_model()`で定義>  
* `self.hyparam_list1`:  
1つ目のハイパーパラメータリスト。`'lasso'`, `'ridge'`, `'elastic'` は`alpha`, `'rf'` は`num_estimators`
* `self.hyparam_list2`:  
2つ目のハイパーパラメータリスト。`'elastic'` は`l1_ratio`, `'rf'` は`max_depth`
* `self.hyparam_array`:  
ハイパーパラメータ2つの時のみ定義する。`self.hyparam_list`, `self.hyparam_list2`の組み合わせのベクトル。
* `self.valid_models`:  
validationに使うモデルの3次元配列。(num_decrease, predict_period, hyparam_list) ハイパーパラメータ2つの時は、3次元 (num_decrease, predict_period, hyparam_list, hyparam_list2) を4次元 (num_decrease, predict_period, hyparam_list * hyparam_list2)に変形して扱っている。  
###### <`increase_validation()`で定義>  
* `self.y_valid_hat`:  
検証での予測値系列。(num_decrease, predict_period, len(hyparam_list))
* `self.valid_error`:  
検証での予測誤差。(len(hyparam_list), num_decrease, predict_period)
* `self.valid_mse`:  
検証でのmse。各predict_period, hyparamに関して、num_decreaseで平均をとる。 各要素は0次元スカラーのxarray.DataArray
* `self.valid_rmse`:  
検証でのrmse。
* `self.best_tuned_model`:  
各predict_periodそれぞれに対してmseを最小化させるモデル。
* `self.best_hyparam`:  
各predict_periodそれぞれに対してmseを最小化させるモデルのハイパーパラメータ。
* `self.best_y_hat`:  
各predict_periodそれぞれに対して最小なmseとなる予測値。
* `self.bset_error`:  
各predict_periodそれぞれに対して最小な予測誤差。
* `self.best_mse`:  
各predict_periodそれぞれに対して最小なmse。
* `self.best_rmse`:  
各predict_periodそれぞれに対して最小なrmse。
* `self.best_hyparam_index`:  
それぞれのpredict_periodのmseを最小化させるハイパーパラメータがhyparam_list(2つの時はかけてベクトルに直したもの)のどこかを示す。  
###### <`set_increase_test_model()`で定義>
* `self.test_models`:  
テスト(最終fit)に使うモデルの1次元配列。(sklearnのオブジェクト)  
###### <`increase_test()`で定義>
* `self.y_test_hat`:  
テストの予測値のarray
* `self.test_error`:  
テストの予測誤差のarray
* `self.test_mse`:  
テストのmse
* `self.test_rmse`:  
テストのrmse
##### Example
`pseudo_dataset15 = Dataset(setting='pseudo', predict_period=30, valid_size=12, days_before=15) # Datasetオブジェクトの定義  
pseudo_dataset15.build() # Datasetオブジェクトの作成  
hyparam_list1 = np.linspace(-10, 10, 21)  
hyparam_list1 = list(10 ** hyparam_list1) # ハイパーパラメータリストの定義  
pseudo_ridge15 = Model(method='ridge', dataset=pseudo_dataset15) # Modelオブジェクトの定義  
pseudo_ridge15.execute(hpl1=hyparam_list1) # Modelの検証および予測の実行  `

### visualize.py
`realnowcast.Model`における検証の様子を3Dプロットする関数を定義。

### masterdata_data_xarray.pkl
OECD.statsより入手したリアルタイムデータを編集して、xarray.Datasetオブジェクトとして保存した。
