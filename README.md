機械学習を用いた日本のGDPナウキャスト
====
## データの発表およびナウキャストのタイミング
![スライド1](https://user-images.githubusercontent.com/59826800/72252220-40230600-3642-11ea-9721-88d61c0d7fef.PNG)

## 疑似コード
### 疑似ナウキャスト

![alg1](https://user-images.githubusercontent.com/59720853/72235602-6c289200-3616-11ea-9360-b869d0dff561.png)

### 予測時点版検証を用いた真正ナウキャスト



### 1次速報値データ検証を用いた真正ナウキャスト

![alg3](https://user-images.githubusercontent.com/59720853/72235631-95e1b900-3616-11ea-927d-0c88de692dfd.png)



## ファイルの概要
### main.py
realnowcast.pyで定義したクラスを用いて、15日前、45日前、75日前の疑似・真正ナウキャストを実行する。realnowcast.Mode.show()でプリントされる結果が表示される。ナウキャストの前に検証(バリデーション)で用いるハイパーパラメータのリストを与える。各検証の様子をvisualize.pyで定義した関数を用いて図示する。

### realnowcast.py
#### class realnowcast.Dataset(data="master_data_xarray.pkl", setting=None, validation=None, predict_period=30, valid_size=12, days_before=15)
GDPナウキャストを行うために使うデータセット。ナウキャストのタイミングと設定によって異なる。
##### Parameters
* data  
用いる元データ。デフォルトでmasterdata_data_xarray.pklを読み込む。
* setting  
疑似ナウキャスト用データセットの場合は'pesudo'、真正ナウキャスト用データセットの場合は'full'と入力する。
* validation  
真正ナウキャストにおける検証の戦略。予測時点版検証なら'pseudo'、1次速報値データ検証なら'full'と入力する。
* predict_period
ナウキャスト対象期の数。
* valid_size 
検証データに用いる期の数。
* days_before
何日前ナウキャストかを指定。

##### Methods
* get_available_data(self, start_period='1994-Q1')  
用いるデータの開始時点を指定して抽出する。
* pseudo_extract_vintage(self)  
疑似ナウキャスト用。self.vintageを参照して1シート抽出し、self.dataとする。
* pseudo_log_diff(self)  
疑似ナウキャスト用。self.dataを対数差分変換する。サンプルサイズが1つ減る。
* pseudo_add_lag(self, feature='GDP', lag=1)  
疑似ナウキャスト用。特徴量の名前とラグを指定して新しくself.dataに追加します。サンプルサイズが1つ減る。
* pseudo_gen_increase_valid_data(self)  
疑似ナウキャスト用。予測の期間(testもvalidationも同じ長さ)、検証の回数を指定して、検証用のself.X_train, self.y_train, self.X_valid, self.y_validを定義する。
* pseudo_gen_increase_test_data(self)  
疑似ナウキャスト用。検証用データと同じ期間の長さでself.X_final_train, y_final_train, self.X_test, self.y_testを定義する。
* full_log_diff(self)  
真正ナウキャスト用。self.dataを対数差分変換する。サンプルサイズが1つ減る。
* full_add_lag(self, feature='GDP', lag=1)  
真正ナウキャスト用。特徴量の名前とラグを指定して新しくself.dataに追加します。サンプルサイズが1つ減る。
* full_gen_increase_valid_data(self)  
真正ナウキャスト用。予測の期間(testもvalidationも同じ長さ)、検証の回数を指定して、検証用のself.X_train, self.y_train, self.X_valid, self.y_validを定義する。
* full_gen_increase_test_data(self)  
真正ナウキャスト用。検証用データと同じ期間の長さでself.X_final_train, y_final_train, self.X_test, self.y_testを定義する。
* full_fillnan(self)  
真正ナウキャスト用。欠損値の処理を行う。  
* build(self)  
パラメータで指定したsettingごとに上記のメソッドを使い分けてDatasetオブジェクトを作成する。

##### Attributes
<クラス作成時に定義>  
* self.data
* self.setting
* self.validation
* self.predict_period
* self.valid_size
* self.feature_name
* self.vintage
* self.days_before: 何日前予測か。15日前, 45日前, 75日前のいずれか。
<_gen_increase_valid_data()で定義>  
* self.X_valid
* self.y_valid
* self.X_train
* self.y_train
<_gen_increase_test_data()で定義>  
* self.X_test
* self.y_test
* self.X_final_train
* self.y_final_train
<full_gen_increase_valid_data()で定義>  
* self.vintage_list = None # vintageの名前のリスト
* self.preliminary_vintage = None # 速報値が発表されたvintageの名前のリスト
<full_fillnan()で定義>  
* self.firstvalue_array = None

### masterdata_data_xarray.pkl
OECD.statsより入手したリアルタイムデータを編集して、xarray.Datasetオブジェクトとして保存した。
