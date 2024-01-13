# スキルゼミ課題 SS2304-01

## 課題名：DICOMファイルを読み込み，3次元データを保存

氏名：金子慧士

## 開発環境

- MacOS ver Ventura 13.3.1
- Python 3.9.12

## 実行方法

- % python3 DICOM2raw_mhd.py (データのあるフォルダ名) (アウトプットのファイル名)

## 使い方
```
    args = get_from_commandline()
    dicom_dir_path, output_filename = args.dicom_dir_path, args.output_filename
    dcm = DICOMHandler(dicom_dir_path, output_filename)
    sorted_list_dicom_files_path = dcm.get_sorted_list_dicom_files_path()
    header_dicom_dict = dcm.get_info_from_dicom_file(sorted_list_dicom_files_path)
    volume = dcm.stack_dicom_files()

    # Argument is directory name
    writer = MetaImageWriter('../data/HeadCtSample_2022')
    writer.save_as_metaimage(image_data=volume, header_info_dic=header_dicom_dict, save_file_name=output_filename)

```
- 実行する際に指定したフォルダ名に含まれるDICOMファイルを読み込む
- 4行目 :  DICOMファイルのInstanceNumberを確認して，それを昇順にソートする．その処理を行った各ファイルのパスを返す
- 5行目 :  DICOMファイルから以下の情報を取得し，辞書形式で返す
```
ObjectType = Image
NDims = 3
DimSize = 512 512 33
ElementType = MET_SHORT
ElementSpacing = 0.54102 0.54102 5.0
ElementByteOrderMSB = False
ElementDataFile = HeadSample2.raw
```
- 6行目 : 各 DICOMファイルから三次元データを作成
- MetaImageWriterクラスでは，インストラクタの引数に保存先のディレクトリのパスを指定
- save_as_metaimageの引数には，三次元データ，ヘッダー情報（辞書形式），保存ファイル名を指定



