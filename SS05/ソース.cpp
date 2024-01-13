
//
// Skill seminer05
// 氏名：河野由貴子　学生番号：18T0811M 最終更新日：2020/10/28
// 使用エディタ：Visual Studio
//

//実行時引数
// 1 テキストファイルのファイルパス
// 2 イメージファイルのファイルパス
// 3 解像度


#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <math.h>
#include <cctype>

using namespace std;
namespace fs = std::filesystem;


#define MIN16bit -32768 //8ビットの最大値
#define MAX16bit 32767   //8ビットの最小値
#define ARGUMENT 4
#define MAX8bit 255 //8ビットの最大値
#define MIN8bit 0   //8ビットの最小値


template <class T>
class Image3d {
public:


    T* image3d;          //画素値を格納する動的配列

    int _Width3d;          //画像のx方向
    int _Height3d;         //画像のy方向
    int _Depth3d;          //画像のz方向
    float _resoX3d;       //x方向の解像度
    float _resoY3d;       //y方向の解像度
    float _resoZ3d;       //z方向の解像度
    int _WindowLevel3d;    //ウィンドウレベル
    int _WindowWidth3d;    //ウィンドウ幅



    //コンストラクタ
    Image3d(int width, int height, int depth, float resox, float resoy, float resoz, int windowlevel, int windowwidth)
        :_Width3d(width), _Height3d(height), _Depth3d(depth), _resoX3d(resox), _resoY3d(resoy), _resoZ3d(resoz), _WindowLevel3d(windowlevel), _WindowWidth3d(windowwidth)
    {
        image3d = new T[width * height * depth];
    }

    //移譲コンストラクタ    
    Image3d(int width, int height, int depth, float resox, float resoy, float resoz)
        :Image3d(width, height, depth, resox, resoy, resoz, 0, 0) {}

    ~Image3d() {
        //デストラクタ
        delete[] image3d;
    }

};

template<class T>
class Image2d {
public:

    T* image2d;          //画素値を格納する動的配列

    int _Width2d;          //画像の幅
    int _Height2d;         //画像の高さ
    int _WindowLevel2d;    //ウィンドウレベル
    int _WindowWidth2d;    //ウィンドウ幅

    //コンストラクタ
    Image2d(int width, int height, int windowlevel, int windowwidth) :_Width2d(width), _Height2d(height), _WindowLevel2d(windowlevel), _WindowWidth2d(windowwidth)
    {
        image2d = new T[width * height];
    }

    //移譲コンストラクタ    
    Image2d(int width, int height)
        :Image2d(width, height, 0, 0) {}


    ~Image2d() {
        //デストラクタ
        delete[] image2d;
    }

};

class Coordinate3d {
public:
    float X;
    float Y;
    float Z;

    //コンストラクタ
    Coordinate3d(float corX, float corY, float corZ) :X(corX), Y(corY), Z(corZ) {}
    //移譲コンストラクタ 
    Coordinate3d() :Coordinate3d(0, 0, 0) {}

};

class Vector3d {
public:
    float X;        //X座標
    float Y;        //Y座標
    float Z;        //Z座標

    //コンストラクタ
    Vector3d(float vecX, float vecY, float vecZ) :X(vecX), Y(vecY), Z(vecZ) {}
    //移譲コンストラクタ　　
    Vector3d() :Vector3d(0, 0, 0) {}

};

template <class T>
void ShowImage(string Imgfilename, Image2d<T>& img) {

    /*ShowImage関数
    *　説明：ファイルに画像データを出力する関数
    *　引数：ファイル名filename,Image2dクラスオブジェクトimg
    *　戻り値：なし
    */

    ofstream ofsImg(Imgfilename, ios::binary);
    if (!ofsImg) {
        //ファイルが開けなかった場合
        cerr << "File do not exist" << endl;
        exit(0);
    }

    int i = 0;
    while (i < img._Height2d * img._Width2d) {
        ofsImg.write((char*)(img.image2d + i), sizeof(T));
        i++;
    }
    ofsImg.close();
}

void GetData(filesystem::path& path, int& width, int& height, int& depth, float& resox, float& resoy, float& resoz, int& windowlevel, int& windowwidth)
{
    /*GetData関数
    *　説明：ファイルから，画像の幅，高さ，ウィンドウレベル，ウィンドウ幅，データタイプを読み込む関数
    *　引数：ファイルパス名，画像の幅，高さ，ウィンドウレベル，ウィンドウ幅，データタイプを入れる変数の参照
    *　戻り値：なし
    */

    ifstream ifs(path);
    string str;

    if (ifs.fail()) {
        //ファイルが開けなかった場合
        cerr << "txtFile do not exist" << endl;
        exit(0);
    }

    smatch match;
    regex re("(\\w+) = (\\w+\\.*\\w*)");

    while (getline(ifs, str)) {
        if (regex_search(str, match, re)) {
            if (match[1] == "Width") {
                width = stoi(match[2]);
            }
            else if (match[1] == "Height") {
                height = stoi(match[2]);
            }
            else if (match[1] == "Depth") {
                depth = stoi(match[2]);
            }
            else if (match[1] == "resoX") {
                resox = stof(match[2]);
            }
            else if (match[1] == "resoY") {
                resoy = stof(match[2]);
            }
            else if (match[1] == "resoZ") {
                resoz = stof(match[2]);
            }
            else if (match[1] == "Windowlevel") {
                windowlevel = stoi(match[2]);
            }
            else if (match[1] == "Windowwidth") {
                windowwidth = stoi(match[2]);
            }
        }
    }
    ifs.close();
}

template <class T, class U>
void GradationProcessing(Image2d<T>& Orgimg, Image2d<U>& Graimg)
{
    /*GradationProcessing関数
    *　説明：階調処理を行う関数
    *　引数：Image2dクラスオブジェクトOrgimg,Graimg
    *　戻り値：なし
    */

    int min = Orgimg._WindowLevel2d - (Orgimg._WindowWidth2d / 2);    //階調前の最大画素値
    int max = Orgimg._WindowLevel2d + (Orgimg._WindowWidth2d / 2);    //階調前の最小画素値

    int i = 0;  //パラメータ

    while (i < Orgimg._Width2d * Orgimg._Height2d) {
        if (Orgimg.image2d[i] >= max)
            Graimg.image2d[i] = (unsigned char)MAX8bit;
        else if (Orgimg.image2d[i] <= min)
            Graimg.image2d[i] = (unsigned char)MIN8bit;
        else
            Graimg.image2d[i] = (unsigned char)((Orgimg.image2d[i] - min) * MAX8bit / (max - min));
        i++;
    }
}


template <class T>
void GetImage(filesystem::path& path, Image3d<T>& img) {

    /*GetImage関数
    *　説明：ファイルから画像を読み込む関数
    *　引数：ファイルパス名path,Image3dクラスオブジェクトimg
    *　戻り値：なし
    */

    ifstream ifs(path, ios::binary);
    if (!ifs) {
        //ファイルが開けなかった場合
        cerr << "imgFile do not exist" << endl;
        exit(0);
    }
    int i = 0;
    while (!ifs.eof()) {
        ifs.read((char*)(img.image3d + i), sizeof(T));
        i++;
    }
    ifs.close();
}

template<class T>
void Rotation(T* rotationArray, Coordinate3d& Original, Coordinate3d& Rotation, Coordinate3d& Center) {

    /*Rotation関数
    *　説明：座標を中心位置を指定して回転させる関数
    *　引数：回転させる角度での回転行列rotationArray,元の座標，移動後の座標，回転中心の座標を入れるCoorinate3dオブジェクト
    *　戻り値：なし
    */

    Rotation.X = (Original.X - Center.X) * rotationArray[0] + (Original.Y - Center.Y) * rotationArray[1] + (Original.Z - Center.Z) * rotationArray[2];
    Rotation.X += Center.X;

    Rotation.Y = (Original.X - Center.X) * rotationArray[3] + (Original.Y - Center.Y) * rotationArray[4] + (Original.Z - Center.Z) * rotationArray[5];
    Rotation.Y += Center.Y;

    Rotation.Z = (Original.X - Center.X) * rotationArray[6] + (Original.Y - Center.Y) * rotationArray[7] + (Original.Z - Center.Z) * rotationArray[8];
    Rotation.Z += Center.Z;

}

template <class T>
void Mip(Image3d<T>& img, Image2d<T>& GetImg, int resolution,int rotAngleX,int rotAngleY,int rotAngleZ) {

    /*Mip関数
    *　説明：３次元画像からMip画像を出力する関数
    *　引数：三次元画像オブジェクトimg,Mip画像を入れるオブジェクトGetImg,解像度resolution
    *　戻り値：なし
    */

    //三次元画像の中心座標をいれるCoordinate3dオブジェクト    
    Coordinate3d center((img._Width3d * img._resoX3d) / 2, (img._Height3d * img._resoY3d) / 2, (img._Depth3d * img._resoZ3d) / 2);

    float R = sqrt(pow(center.X, 2) + pow(center.Y, 2) + pow(center.Z, 2));   //半径の算出
    float diameter = R * 2; //直径の算出

    //回転前のMip画像の中心座標をいれるCoordinate3dオブジェクト
    Coordinate3d Sqrtcenter(center.X, center.Y - R, center.Z);
    //回転前のMip画像の3つの頂点の座標をいれるCoordinate3dオブジェクト 
    Coordinate3d Sqrtvertex1(center.X - R, center.Y - R, center.Z - R);
    Coordinate3d Sqrtvertex2(center.X - R, center.Y - R, center.Z + R);
    Coordinate3d Sqrtvertex3(center.X + R, center.Y - R, center.Z - R);


    //角度をラジアンに変換する．
    float rad = M_PI / 180;
    float radX, radY, radZ;
    radX = rad * (float)rotAngleX;
    radY = rad * (float)rotAngleY;
    radZ = rad * (float)rotAngleZ;

    //回転のためにサインコサインの計算を行う
    float SinX, SinY, SinZ, CosX, CosY, CosZ;
    SinX = sin(radX);
    SinY = sin(radY);
    SinZ = sin(radZ);
    CosX = cos(radX);
    CosY = cos(radY);
    CosZ = cos(radZ);

    //回転行列を格納するための配列
    float roundArray[9] = { CosZ * CosY, CosZ * SinY * SinX + SinZ * CosX, SinZ * SinX - CosZ * SinY * CosX,
                            -SinZ * CosY, CosZ * CosX - SinZ * SinY * SinX, SinZ * SinY * CosX + CosZ * SinX,
                            SinY,       -CosY * SinX,                      CosY * CosX };

    //回転後のMip画像の中心座標をいれるCoordinate3dオブジェクト
    Coordinate3d rotSqrtcenter;
    Rotation(roundArray, Sqrtcenter, rotSqrtcenter, center);

    //回転後のMip画像の3つの頂点の座標をいれるCoordinate3dオブジェクト
    Coordinate3d rotSqrtvertex1;
    Rotation(roundArray, Sqrtvertex1, rotSqrtvertex1, center);

    Coordinate3d rotSqrtvertex2;
    Rotation(roundArray, Sqrtvertex2, rotSqrtvertex2, center);

    Coordinate3d rotSqrtvertex3;
    Rotation(roundArray, Sqrtvertex3, rotSqrtvertex3, center);

    //中心に向かう単位ベクトル（解像度分）
    Vector3d unitVector((center.X - rotSqrtcenter.X) / R,
        (center.Y - rotSqrtcenter.Y) / R,
        (center.Z - rotSqrtcenter.Z) / R);

    //Mip画像の縦横のベクトル
    //縦
    Vector3d Vertical(rotSqrtvertex2.X - rotSqrtvertex1.X,
        rotSqrtvertex2.Y - rotSqrtvertex1.Y,
        rotSqrtvertex2.Z - rotSqrtvertex1.Z);
    //横
    Vector3d Horizonal(rotSqrtvertex3.X - rotSqrtvertex1.X,
        rotSqrtvertex3.Y - rotSqrtvertex1.Y,
        rotSqrtvertex3.Z - rotSqrtvertex1.Z);

    //Mip画像の縦横の単位ベクトル(解像度分)
    //縦
    Vector3d unitVertical(Vertical.X / resolution,
        Vertical.Y / resolution,
        Vertical.Z / resolution);
    //横
    Vector3d unitHorizonal(Horizonal.X / resolution,
        Horizonal.Y / resolution,
        Horizonal.Z / resolution);

    //Mip画像上を動くパラメータ
    float paramX = rotSqrtvertex1.X,
        paramY = rotSqrtvertex1.Y,
        paramZ = rotSqrtvertex1.Z;

    //動かす光線の座標
    float moveX, moveY, moveZ;

    //3次元画像の平面
    int planeImg = img._Width3d * img._Height3d;


    T max;  //最大値を格納する変数


    int p;  //外接円の端まで行ったかを判断するためのパラメータ

    int _paramImg, _paramX, _paramY, _paramZ; //画像の位置を入れるためのパラメータ


    for (int Height2d = 0; Height2d < resolution; Height2d++, paramX += unitVertical.X, paramY += unitVertical.Y, paramZ += unitVertical.Z) {

 

            for (int Width2d = 0; Width2d < resolution; Width2d++, paramX += unitHorizonal.X, paramY += unitHorizonal.Y, paramZ += unitHorizonal.Z) {
                    //動かす光線の初期位置の格納
                    moveX = paramX, moveY = paramY, moveZ = paramZ;

                //最大値をshort型の最小値にする
                max = MIN16bit;

                //外接円の端まで行ったかを判断するためのパラメータ
                p = 0;

                //3d画像値に入るまでの処理（入らなかった場合はpで判断する）
                while ((0 > moveX || moveX > (2 * center.X) || 0 > moveY || moveY > (2 * center.Y) || 0 > moveZ || moveZ > (2 * center.Z)) && p < diameter) {
                    //単位ベクトル分の光線の移動
                    moveX += unitVector.X;
                    moveY += unitVector.Y;
                    moveZ += unitVector.Z;
                    p++;

                }
                //入らなかった場合はforループを抜ける
                if (p == diameter) {
                    continue;
                }

                //3d画像に入った場合の処理（でたら抜ける）
                while ((0 <= moveX && moveX < (2 * center.X)) && (0 <= moveY && moveY < (2 * center.Y)) && (0 <= moveZ && moveZ < (2 * center.Z))) {


                    _paramX = (int)(moveX / img._resoX3d + 0.5);
                    _paramY = (int)(moveY / img._resoY3d + 0.5);
                    _paramZ = (int)(moveZ / img._resoZ3d + 0.5);

                    //画像の幅を超えたら一番近い値で置き換える 
                    if (_paramX >= img._Width3d) {
                        _paramX = img._Width3d - 1;
                    }
                    if (_paramY >= img._Height3d) {
                        _paramY = img._Height3d - 1;
                    }
                    if (_paramZ >= img._Depth3d) {
                        _paramZ = img._Depth3d - 1;
                    }

                    //光線位置の画像の座標の計算
                    _paramImg = _paramZ * planeImg + _paramY * img._Width3d + _paramX;

                    //最大値を更新したら置き換える
                    if (max < img.image3d[_paramImg]) {
                        max = img.image3d[_paramImg];
                    }

                    //単位ベクトル分の光線の移動
                    moveX += unitVector.X;
                    moveY += unitVector.Y;
                    moveZ += unitVector.Z;
                }


                //Mip画像に最大値を格納
                GetImg.image2d[Height2d * resolution + Width2d] = max;

            }


        //Mip画像のパラメータを戻す．
        paramX -= Horizonal.X;
        paramY -= Horizonal.Y;
        paramZ -= Horizonal.Z;

    }

}

bool check_int(string str)
{
    /*
        check_int関数
        説明：取得した文字列が数字かを判断する関数
        引数：string型変数
        戻り値：True or False

    */
    if (all_of(str.cbegin(), str.cend(), isdigit))
    {
        return true;
    }
        cout << "int型整数でもう一度入力してください" << endl;
    return false;
}

int main(int argc, char* argv[])
{
    //引数の個数が一致しなかった場合は終了
    if (argc != ARGUMENT) {
        cout << "Argument is not appropriate" << endl;
        return -1;
    }

    //パス名を絶対パスに変換
    fs::path abs_p1{ argv[1] };
    fs::path rel_p1 = fs::relative(abs_p1);

    //ファイルが存在しなかったら終了
    if (!fs::exists(rel_p1)) {
        std::cout << ".....No txtFile exist....." << std::endl;
        return -1;
    }

    //データを入れる変数の定義
    int width, height, depth, windowlevel, windowwidth;
    float resox, resoy, resoz;

    //データをテキストファイルから抽出する
    GetData(rel_p1, width, height, depth, resox, resoy, resoz, windowlevel, windowwidth);

    //パス名を絶対パスに変換
    fs::path abs_p2{ argv[2] };
    fs::path rel_p2 = fs::relative(abs_p2);

    //ファイルが存在しなかったら終了
    if (!fs::exists(rel_p2)) {
        std::cout << ".....No imgFile exist....." << std::endl;
        return -1;
    }

    //画像を入れるImage3dオブジェクトの作成
    Image3d<short> OrgImg(width, height, depth, resox, resoy, resoz, windowlevel, windowwidth);

    //画像データをオブジェクトに格納する
    GetImage(rel_p2, OrgImg);

    //実行時引数から解像度の取得
    int resolution = atoi(argv[3]);

    //Mip画像を入れるImage2dオブジェクトの作成
    Image2d<short> MipImg(resolution, resolution, OrgImg._WindowLevel3d, OrgImg._WindowWidth3d);

    string strAngle;
    int rotationangleX, rotationangleY, rotationangleZ;

    do{

        cout << "それぞれの方向の回転角度入力(角度で入力して下さい)\n正面が初期位置です\nx:";
        cin >> strAngle;

    } while (!check_int(strAngle));

    rotationangleX = stoi(strAngle);

    do {

        cout << "y:";
        cin >> strAngle;

    } while (!check_int(strAngle));

    rotationangleY = stoi(strAngle);

    do {

        cout << "z:";
        cin >> strAngle;

    } while (!check_int(strAngle));

    rotationangleZ = stoi(strAngle);

    //Mip処理
    Mip(OrgImg, MipImg, resolution,rotationangleX,rotationangleY,rotationangleZ);

    //諧調処理後の画像を入れるImage2dオブジェクトの作成
    Image2d<unsigned char> MipGraImg(MipImg._Width2d, MipImg._Height2d);

    //諧調処理を行う
    GradationProcessing(MipImg, MipGraImg);

    //Mip画像の出力
    ShowImage("mip.raw", MipGraImg);

}