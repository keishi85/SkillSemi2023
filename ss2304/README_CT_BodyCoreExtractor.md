# スキルゼミ課題 SS230X-02

## 課題名：CT画像の体幹抽出

氏名：金子慧士

## 開発環境

- MacOS ver Ventura 13.3.1


## 実行方法

- % python3 CT_BodyCoreExtractor.py



### 使い方
```CT_data_dir_path = '../data/ChestCT/ChestCT.mhd'
    ct_3d = CT3DImageProcesser(CT_data_dir_path)
    thresholded_3D_image = ct_3d.thresholding_3D_image(ct_3d.image_array, -100, 200)
    morphology_3D_image = ct_3d.morphological_3D_operations(thresholded_3D_image)
    labeled_3D_img = ct_3d.larger_connected_component3D(morphology_3D_image)

    # Read mhd file
    data_read = MetaDataRead('../data/ChestCT/ChestCT.mhd')
    mhd_dict = data_read.read_as_dict()
    # Changing the output file name and element type
    mhd_dict['ElementDataFile'] = 'ss2304-02.raw'

    # Write raw file
    writer = MetaImageWriter('../data/ChestCT')
    # Change 3D_img to property data type
    data_type = writer.get_data_type(mhd_dict['ElementType'])
    writer.save_as_metaimage(image_data=labeled_3D_img.astype(data_type), header_info_dic=mhd_dict, save_file_name='ss2304-02')
   ```
- CT3DImageProcesserのインストラクタの引数にはCT画像が含まれるディレクトリのパスが入る
- 以下は CT3DImageProcesserのメソッドの説明である
- thresholding_3D_image : 画像の二値化
- morphological_3D_operations : モルフォロジー処理
- larger_connected_component3D : ラベリングを行う
- 以上で画像処理は終了である
- mhdファイルに書き込む際には，ファイル名とMeraImageWriterクラスのget_data_typeから得たデータタイプの変換を行う
- その後，mhd, rawファイルを作成


