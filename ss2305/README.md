# スキルゼミ課題 SS2305

## 課題名：

氏名：金子慧士

## 開発環境

- MacOS ver Ventura 13.3.1
- (Clang 14.0.3)

## 実行方法

- % python kfold_train.py

### 準備データ
以下の場所からデータを取得してください
学習済みモデルもあります．
保存先は以下のフォルダ構造を参照してください
```
\\gabor\Free\金子
```

以下は元データの場所です．
````
症例データ：¥¥gabor¥Data¥Cardiology¥西毅先生20230925¥CHIBAMI_No_reflow_prediction¥list¥
症例情報：¥¥gabor¥Data¥Cardiology¥西毅先生20230925¥CHIBAMI_No_reflow_prediction¥CHIBAMI_case_list.xlsx
```


### 使い方(各関数の使い方は以下を参照)
1. zipファイルの解凍(unfreeze_zip.py)
2. IVUS, negative, positiveフォルダを以下のファイル構造となるように作成(create_dataset.py)
3. kfold_train.py, kfold_not_over_sampling.pyにて実行
   (おそらく，kfold_train.pyが一番精度が良いと思われます)


### ファイル構造
``` 
ss2305 
    |- data 
        |- list
        |- IVUS   
            |- negative
                |- ex1.png
                ...
            |- positive
                |- ex2.png
                ... 
        |-   .xml
        
                  -  
    |- src 
        |- kfold_train.py
        |- kfold_not_over_sampling.py  
        |- utils   
            |- XmlProcessor.py
            |- unfreeze_zip.py
            |- XmlProcessor.py
            |- create_dataset.py
            |- Dataset.py
        |- models
            |- resenet18.py
           
```
## kfold_train.py の使い方
上記のようなフォルダ構成にしてください．
以下のようにmain関数を定義します．モデルは'GoogleNet', 'ResNet18', 'vgg16'の中から選べます．
5 fold cross validationが実装されています．
'GoogleNet', 'VGG16'は事前学習済みモデルを採用しています．
Customデータセットでは'negative'クラスのデータを'[90, 180, 270]'に回転させ，データ数を4倍にしています．（'[90, 180]'にすれば，3倍のデータ数になります）
テストデータに分割されたデータから症例ごとに最小フレームの値が入力として与えられます．
訓練プロセスに関して，各foldでの損失と学習曲線，roc曲線などを保存します．
```
if __name__ == "__main__":
    data_dir = '../data/IVUS'
    output_path = '../data/result'
    lr = 0.001
    batch_size = 16
    max_epoch = 50

    # 'ResNet', 'GoogleNet', 'ResNet18', 'vgg16'
    ave_val_acc = train(data_dir, model_name='GoogleNet', output_path=output_path,
                        learning_rate=lr,
                        class_weight=True,
                        batch_size=batch_size,
                        max_epoch=max_epoch) 
```


### XmlProccesor.py
コンストラクタでは引数は不要です．
### load_xml()
- 引数 : xmlファイルがあるディレクトリ名
- 返り値 : ディレクトリ内の.xmlファイルのリスト
- もし，存在しないディレクトリ名を入力した際にはエラーが発生

### anonymize_xml() 
- 引数 : xmlファイルのリスト，保存先ファイルパス
- xmlファイル内の 'PatientName', 'PatientBirthDateの値を '*' に書き換える
```
    dir_path = '../../data/list'
    preserve_path = '../../data/anonymized_list'

    xml = XmlProcessor()
    xml.anonymize_xml(dir_path, preserve_path)
```

### unfreeze_zip.py
- 引数 : 入力ディレクトリのパス，zipファイルの展開先のパス
- 入力ディレクトリ内のzipファイルを展開する
```
    unfreeze_zip('../../data/anonymized_list', '../../data/anonymized_list')
```

### create_dataset.py
#### CreateLabel
- インスタンスの引数 : (それぞれのフォルダがある親ディレクトリ, ラベルが書かれたxlsxファイル)
- extract_img() : 画像データのパスをすべて取得し, リストで返す
- extract_data_from_excel() : excelファイルから、key : 'ID', value : 'Final_TIMI'のデータを返す
- extract_no_from_path() : 引数 : 画像データのパス, key : すべての画像データのパス, value : ラベル　の辞書形式で返す
- dived_image() : 引数 : 上記二つの返り値，保存先のパス，　画像データをラベルごとに異なるフォルダに保存する, 返り値には'No'とそれが'negative', 'positive'かの辞書である
- ex) 'CHIBAMI_67.1' のディレクトリ内の画像はすべて 'frame_1980_67.1.png' に変換されてラベルごとのディレクトリに保存される
```
img_dir_path = '../../data/anonymized_list'
    xml_path = '../../data/CHIBAMI_case_list.xlsx'

    cd = CreateLabel(img_dir_path, xml_path)

    # data_dict key : 'No', value : 'Final_TIMI'
    img_path_list = cd.extract_img()
    data_dic = cd.extract_data_from_excel()

    # Extract 'No' from image path
    path_No_dic = cd.extract_no_from_path(img_path_list)

    # Diving image depend on 'Final_TIMI'
    save_path = '../../data/VIUS'
    cd.dived_image(data_dic, path_No_dic, save_path)
```
 
### Dataset.py
- データ拡張行うためのクラス
- コード内の '[90, 180, 270]' の回転がされるため，元画像より4倍多くなる
```
        dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])

        # Divide dataset to train_dataset and test_dataset
        train_val_index, test_index = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=seed)
        # Convert training and validation set indices to actual indices
        train_index = train_val_index[train_index]
        val_index = train_val_index[val_index]

        # The train dataset rotates the 'negative' class image data by 90, 180, and 270 degrees,
        # increasing the number of images by a factor of 4.
        train_dataset = torch.utils.data.dataset.Subset(dataset, train_index)
        train_dataset = CustomDataset(root=data_dir, indices=train_dataset.indices, transform=data_transforms['train'],)

        val_dataset = torch.utils.data.dataset.Subset(dataset, val_index)
        test_dataset = torch.utils.data.dataset.Subset(dataset, test_index)
```

### 特徴
- 5 fold cross validationを実装
- オーバーサンプリングの手法として90, 180, 270度変換させたデータを'negative'クラスに追加（つまり，データ数が4倍になる）
- クラスごとの重みづけを実装
- model_nameからVGG16, ResNet18, GoogleNetを選択可能

### 注意
- フォルダ等が自動で生成されない場合があるので，エラーが生じる際にはフォルダを事前に生成して試してください．

