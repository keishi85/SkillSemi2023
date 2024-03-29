# スキルゼミ課題 SS2307

## 課題名： C++ 3次元画像の補間

氏名：金子慧士

## 開発環境

- MacOS ver Ventura 13.3.1
- (Clang 14.0.3)

## 実行方法

- % g++ -O2 -std=c++14 main.cpp && ./a.out

### mhdRaw.hpp
mhd, rawファイルの読み書きを行うクラスが適されています


### 使い方
このプログラムは，医用画像の等方化をトリキュービック補間を使用して実行します
入力として, MHDファイルとRAWファイルのパスを指定します
出力として, 等方化されたRAWファイルとMHDファイルが生成されます


### 特徴
元の画像データに対してZ軸の解像度をX軸およびY軸の解像度に合わせて調整します
トリキュービック補間により, Z軸方向の新しいボクセル値を計算し, 元の画像の解像度を維持しつつ, Z軸を等方化します
等方化された画像は, 元の画像の詳細を保持しつつ, Z軸の解像度を向上させます


### 注意
入力ファイルはMHD/RAW形式である必要があります．他の形式はサポートされていません．
出力ファイルは元のファイルと同じディレクトリに保存されます．既存のファイルと名前が重複しないように注意してください．

##　参考資料
https://imagingsolution.blog.fc2.com/blog-entry-142.html

