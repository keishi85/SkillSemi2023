# スキルゼミ課題 SS2306

## 課題名：画像フィルタリング

氏名：金子慧士

## 開発環境

- MacOS ver Ventura 13.3.1
- (Clang 14.0.3)

## 実行方法

- % g++ -O2 -std=c++20 main.cpp && ./a.out



## ImageProcessor
ImageProcessorはRAW形式の異様画像データの読み込み，ウィンドウレベリング，フィルタリング処理を行うことができる．また，処理後のデータを保存可能．

## 機能
- AW画像データの読み込み
- ウィンドウレベリングによる画像の階調調整
- ソーベルフィルタによるエッジ検出
- 移動平均フィルタによる画像の平滑化
- メディアンフィルタによるノイズ除去
- 処理後の画像データの保存

## 使い方
- インスタンスの生成時にコマンドラインからテキストファイルのパスを入力(テキストファイルの内容については以下を参照)
- テキストファイルからwindowlevelingを行う場合，getUnsinedBuffer(), dataTyepe="MET_UCHAR"を指定
- windowlevelingを行わない場合，getShortBuffer(), dataType="MET_SHORT"を指定
```
int main(){
    // std::string path = "./data/ProcessingParameter2.txt";
    ImageProcessor processor; // Get text data and mhd data
    processor.loadRawImage();
    processor.windowLeveling();


    // If WindowProcessing is False, 
    // std::vector<short> buffer = processor.getShortBuffer();
    // dataType = "MET_SHORT"
    std::vector<unsigned char> buffer = processor.getUnsignedBuffer();
    std::string dataType = "MET_UCHAR";

    // Filter processing 
    buffer = processor.filterProcess(buffer);
    
    std::string savePath = "./data";
    processor.saveRawImage(buffer, savePath);
    processor.createMhdFile(dataType, savePath);

}
```

## 注意
loadRawImage()におけるエラーの内容
```
return 1 : ファイルでのエラー 
return 2 : 他のエラー
```

mhdファイルの以下はデフォルトで指定
```
ElementSpacing:
mhdFile << "ElementSpacing = 1.000000 1.000000\n";
画像の各ピクセル（またはボクセル）間の物理的な間隔を指定．ここでは，両方の方向（通常はX軸とY軸）で1.0と指定．これは，ピクセル間の距離が1.0の単位（例えば1mm，1cmなど，実際の単位は画像のコンテキストに依存する）であることを意味

ElementByteOrderMSB:
mhdFile << "ElementByteOrderMSB = False\n";
この行では，画像データのバイト順（エンディアン）を指定．False はリトルエンディアンを意味し，各ピクセル値の最も重要でないバイト（Least Significant Byte、LSB）が先に格納されることを示す．逆に True ならばビッグエンディアンを意味し，最も重要なバイト（Most Significant Byte，MSB）が先に格納されることを示す．
```

### テキストファイルについて
- ImageProcessing = ["SobelX", "SobelY", "MovingAverage", "Median"]から指定
```
Input = CT_Noise
Output = CT_Noise_Median
ImageProcessing = Median
WindowProcessing = True
WindowLevel = 100
WindowWidth = 400
MovingAverageFilterKernel = 3
MedianFilterKernel = 3
```