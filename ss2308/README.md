# スキルゼミ課題 SS2308

## 課題名：3次元画像投影方法と高速化

氏名：金子慧士

## 開発環境

- MacOS ver Ventura 13.3.1
- (Clang 14.0.3)

## 実行方法

- % g++ -O2 -std=c++20 main.cpp && ./a.out

## コメント
Mip画像がうまく生成できていないです．
何が原因かわかっていません．



### 使い方
このプログラムは3D CT画像からMIP画像を生成するためのものです．プログラムは与えられた3D画像データを読み込み，画像内の各点を回転行列を用いて変換し，MIP画像を生成します．生成されたMIP画像は，指定されたファイルパスに保存されます．


### 特徴
画像のウィンドウレベリングを行い、視覚化を改善します
3D画像を回転させて、異なる角度からのMIP画像を生成できます


### 注意
プログラムの実行には3D画像データ（.mhdおよび.rawファイル）が必要です
パラメータファイル（ProcessingParameter.txt）には，ウィンドウレベリングの設定や回転角度などが記載されている必要があります