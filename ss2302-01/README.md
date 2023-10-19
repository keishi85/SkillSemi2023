# スキルゼミ課題 SS2302-01

## 課題名： グレースケール画像のcropping / padding 

氏名：金子慧士

## 開発環境

- MacOS ver Ventura 13.3.1

## 実行方法

- python3 image_resizer.py (input_filename) (output_filename) (output_pixel_size: tuple) (base_point) 

## コメント

### 使い方

クラスImageProcessのインスタンスでは引数に
- 入力画像ファイル名
- 出力画像ファイル名
を取得する

メソッドresize()の引数では
- 出力画像サイズ
- cropping / padding の基点(0-4)を指定(デフォルトは0)
- 0: 中央
- 1: 左上
- 2: 左下
- 3: 右下
- 4: 右上

### 特徴

paddingでは黒い背景が追加させる．



