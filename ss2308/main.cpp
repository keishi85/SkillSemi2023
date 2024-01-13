#include <iostream>
#include <vector>
#include <map>
#include <array>

#include "./mhdRaw.hpp"
// #include "./Matrix.hpp"

using namespace std;
using Mat9 = array<array<double, 3>, 3>;
using Vector3 = array<double, 3>; 

#define MIN16bit -32767 // 

struct Vector3d{
    double x; 
    double y;
    double z;
};

class Mip : LoadMhdRaw{
    map<string, string> parameter;
    map<string, string> mhd;
    int width, height, depth;
    int dimSize;
    float xResolution, yResolution, zResolution; //解像度
    vector<short> shortBuffer;
    vector<unsigned char>ucharBuffer;

    float centerX, centerY, centerZ;
    float radius;
    vector<unsigned char> mipImage;

    float roundArray[9];    // 回転行列　

public:
    void loadParameter(string path){parameter = loadText(path);}
    void loadMhd(string path){mhd = loadmhd(path);}
    void loadRaw(string path);
    void windowLeveling();

    template<typename T>
    void saveRaw(const std::vector<T>& buffer, const std::string savePath){saveRawImage(buffer, savePath);}

    void saveMhd(const std::string savePath);

    vector<unsigned char>getUnsingedBuffer(){return this->ucharBuffer;}
    vector<short>getShortBuffer(){return this->shortBuffer;}
    vector<unsigned char> getMip(){return this->mipImage;}

    // Get value from x, y, z axis
    template<typename T>
    T getVoxelValue(T buffer, int x, int y, int z){return buffer[x + y * width + z * width * height];}

    float calcRadius();
    void generateImage(float radius);

    // 行列の計算
    Vector3 multiplyMatrixVector(const Mat9& matrix, const Vector3& vector);
    Mat9 multiplyMatrixMatrix(const Mat9& matrix1, const Mat9& matrix2);

    // 回転行列の計算　
    void calcRotationMatrix(float roll, float pitch, float yaw);

    // 実際に三次元座標を回転させる関数
    void rotate(Vector3d original, Vector3d &rotated, Vector3d center);

    void MipCreate();
};

void Mip::loadRaw(string path){
    if(mhd.find("DimSize") != mhd.end()){
        std::string dimSize = mhd["DimSize"];
        std::istringstream iss(dimSize);
        iss >> width >> height >> depth;
    }

    shortBuffer = loadRawImage(path, width, height, depth);
}

void Mip::windowLeveling(){
    // Finish if WindowProcessing != True
    if(this->parameter["WindowProcessing"] != "True"){
        std::cout << "No gradation conversion." << std::endl;
        return;
    }

    short windowWidth = stoi(this->parameter["WindowWidth"]);
    short windowCenter = stoi(this->parameter["WindowLevel"]);

    short windowMin = windowCenter - windowWidth / 2;
    short windowMax = windowCenter + windowWidth / 2;

    this->ucharBuffer.clear();
    this->ucharBuffer.resize(this->shortBuffer.size());

    for(size_t i = 0; i < this->shortBuffer.size(); i++){
        short pixelValue = this->shortBuffer[i];

        if(pixelValue < windowMin){
            ucharBuffer[i] = 0;
        }
        else if(pixelValue > windowMax){
            ucharBuffer[i] = 255;
        }
        else{
            ucharBuffer[i] = static_cast<unsigned char>(255.0 * (static_cast<float>(pixelValue - windowMin) / windowWidth));
        }
    }
}

void Mip::saveMhd(const std::string savePath){
    if (savePath.substr(savePath.find_last_of(".") + 1) != "mhd") {
        std::cout << "拡張子を\'.mhd\' にしてください．" << std::endl;
        return;
    } 

    // If type of buffer is short, data type is "MET_SHORT"
    // Else data type is "MET_UCHAR"
    std::string dataType;
    if (this->parameter["WindowProcessing"] != "True") {
        dataType = "MET_SHORT";
    } 
    else {
        dataType = "MET_UCHAR";
    }

    std::ofstream mhdFile(savePath);

    std::string rawFileName = LoadMhdRaw::getFileNameFromPath(savePath) + ".raw";

    if(width == 0 || height == 0){
        std::cout << "width, height が定義されていません. \nLoadRawImageを実行してください." << std::endl;
        return;
    }

    mhdFile << "ObjectType = Image\n";
    mhdFile << "NDims = 2\n";
    mhdFile << "DimSize = " << dimSize << " " << dimSize << "\n";
    mhdFile << "ElementType = " << dataType << "\n";
    mhdFile << "ElementSpacing = " << xResolution << " " << yResolution << "\n";
    mhdFile << "ElementByteOrderMSB = False\n";
    mhdFile << "ElementDataFile = " << rawFileName << "\n";
}

float Mip::calcRadius(){
    if(mhd.find("ElementSpacing") != mhd.end()){
        std::string dimSize = mhd["ElementSpacing"];
        std::istringstream iss(dimSize);
        iss >> xResolution >> yResolution >> zResolution;
    }
    else{
        cout << "ElementSpacing は定義されていません．" << endl;
        return 0;
    }

    float xLength = width * xResolution;
    float yLength = height * yResolution;
    float zLength = depth * zResolution;

    centerX = xLength / 2.0;
    centerY = yLength / 2.0;
    centerZ = zLength / 2.0;

    radius = sqrt(centerX * centerX + centerY * centerY + centerZ * centerZ);

    dimSize = static_cast<int>(2 * radius);

    return radius;
}

void Mip::generateImage(float radius){
    mipImage.resize(2 * radius * 2 * radius);
}

Vector3 Mip::multiplyMatrixVector(const Mat9& matrix, const Vector3& vector) {
    Vector3 result{};
    for (int i = 0; i < 3; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < 3; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

Mat9 Mip::multiplyMatrixMatrix(const Mat9& matrix1, const Mat9& matrix2) {
        Mat9 result{};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result[i][j] = 0.0;
                for (int k = 0; k < 3; ++k) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return result;
}

void Mip::calcRotationMatrix(float roll, float pitch, float yaw){
    // 角度をラジアンに変換する
    float rad = M_PI / 180;
    float radX, radY, radZ;
    radX = rad * roll;
    radY = rad * pitch;
    radZ = rad * yaw;

    float sinX, sinY, sinZ, cosX, cosY, cosZ; 
    sinX = sin(radX);
    sinY = sin(radY);
    sinZ = sin(radZ);
    cosX = cos(radX);
    cosY = cos(radY);
    cosZ = cos(radZ);

    //回転行列を格納するための配列
    this->roundArray[0] = cosZ * cosY;
    this->roundArray[1] = cosZ * sinY * sinX + sinZ * cosX;
    this->roundArray[2] = sinZ * sinX - cosZ * sinY * cosX;
    this->roundArray[3] = -sinZ * cosY;
    this->roundArray[4] = cosZ * cosX - sinZ * sinY * sinX;
    this->roundArray[5] = sinZ * sinY * cosX + cosZ * sinX;
    this->roundArray[6] = sinY;
    this->roundArray[7] = -cosY * sinX;
    this->roundArray[8] = cosY * cosX;
    return;
}

void Mip::rotate(Vector3d original, Vector3d &rotated, Vector3d center){
    rotated.x = (original.x - center.x)  * roundArray[0] + (original.y - center.y) * roundArray[1] + (original.z - center.z) * roundArray[2];
    rotated.x += center.x;

    rotated.y = (original.x - center.x) * roundArray[3] + (original.y - center.y) * roundArray[4] + (original.z - center.z) * roundArray[5];
    rotated.y += center.y;

    rotated.z = (original.x - center.x)* roundArray[6] + (original.y - center.y) * roundArray[7] + (original.z - center.z) * roundArray[8];
    rotated.z += center.z;
}

void Mip::MipCreate(){

    // 3次元画像に接する級の半径を定義
    float r = Mip::calcRadius();
    float diameter = r * 2;
    cout << "radius =" << r << endl;

    // 3次元画像の中心座標　[mm]    centerX[mm]
    Vector3d center = {centerX, centerY, centerZ};

    // 回転前のmip imageの中心座標[mm]
    Vector3d mipCenter{center.x, center.y - r, center.z};

    // 回転前のmip imageにおける3点の画像を算出[mm]
    Vector3d vertex1{center.x - r, center.y - r, center.z - r};
    Vector3d vertex2{center.x - r, center.y - r, center.z + r};
    Vector3d vertex3{center.x + r, center.y - r, center.z - r};

    //  テキストファイルから回転角度を取得 
    float angleX, angleY, angleZ;
    if(parameter.find("ViewAngle") != parameter.end()){
        std::string angle = parameter["ViewAngle"];
        std::istringstream iss(angle);
        iss >> angleX >> angleY >> angleZ;
    }
    
    // 回転行列の計算
    Mip::calcRotationMatrix(angleX, angleY, angleZ);
    
    // Mip imageを回転させた時の中心座標と各頂点を求める
    Vector3d rotMipCenter;
    Mip::rotate(mipCenter, rotMipCenter, center);

    Vector3d rotVertex1;
    Mip::rotate(vertex1, rotVertex1, center);

    Vector3d rotVertex2;
    Mip::rotate(vertex2, rotVertex2, center);

    Vector3d rotVertex3;
    Mip::rotate(vertex3, rotVertex3, center);

    /// Todo : 改良すべきかも？　 'r' で割るのが正しいかな？
    // 単位ベクトルを計算(解像度[mm])
    Vector3d unitCenter{
        (center.x - rotMipCenter.x) / r,
        (center.y - rotMipCenter.y) / r,
        (center.z - rotMipCenter.z) / r
    };

    Vector3d vertical{
        rotVertex2.x - rotVertex1.x,
        rotVertex2.y - rotVertex1.y,
        rotVertex2.z - rotVertex1.z
    };

    Vector3d horizotal{
        rotVertex3.x - rotVertex1.x,
        rotVertex3.y - rotVertex1.y,
        rotVertex3.z - rotVertex1.z
    };

    // diameter =  3次元画像における縦と横
    Vector3d unitVertical{
        (rotVertex2.x - rotVertex1.x) / diameter,
        (rotVertex2.y - rotVertex1.y) / diameter,
        (rotVertex2.z - rotVertex1.z) / diameter
    };

    Vector3d unitHorizonal{
        (rotVertex3.x - rotVertex1.x) / diameter,
        (rotVertex3.y - rotVertex1.y) / diameter,
        (rotVertex3.z - rotVertex1.z) / diameter
    };

    // Todo : もっと早くなる方法ありそう, 以下のプログラムではrayに加えて言ってるけど，　引かないといけない場合もある
    // mip imageの生成
    this->mipImage.resize(dimSize * dimSize, 0);

    // mip imageにおける x, y, z
    double mipX = rotVertex1.x, mipY = rotVertex1.y, mipZ = rotVertex1.z;

    // 光線の位置座標
    double rayX = mipX, rayY = mipY, rayZ = mipZ;

    cout << "unitCenter: " << unitCenter.x <<  ", " << unitCenter.y << ", " <<  unitCenter.z <<  endl;
    cout << "unitVertical: " << unitVertical.x <<  ", " << unitVertical.y << ", "<< unitVertical.z  <<  endl;
    cout << "unitHorizontal: " << unitHorizonal.x <<  ", " << unitHorizonal.y << ", "<< unitHorizonal.z  <<  endl;
    cout << "center: " << centerX  <<  ", " << centerY << ", "<< centerZ  <<  endl;
    cout << "vector: " << unitHorizonal.x <<  ", " << unitHorizonal.y << ", "<< unitHorizonal.z  <<  endl;
    cout << "mip : " << mipX << ", " << mipY << ", " << mipZ << endl;
    cout << "ray :" << rayX << ", " << rayY << ", " << rayZ << endl;
    cout << "ucharBuffer size :" << ucharBuffer.size() << endl;

    // x, y は mip image おける座標
    for(int y = 0; y < diameter; y++, mipX += unitVertical.x, mipY += unitVertical.y, mipZ += unitVertical.z){
        for(int x = 0; x < diameter; x++, mipX += unitHorizonal.x, mipY += unitHorizonal.y, mipZ += unitHorizonal.z){
            // 光線の初期位置を設定
            rayX = mipX;
            rayY = mipY;
            rayZ = mipZ;

            unsigned char max = 0;  // 最大値を格納
            
            // Todo : 直径ないか判断する必要あり？
            // 3次元画像内に入るまで光線を進める
            int count = 0;
            while((rayX < 0 || rayX > (2 * center.x) || rayY < 0 || rayY > (2 * center.y) || rayZ < 0 || rayZ > (2 * center.z)) && count < diameter){
                // 光線を単位ベクトル分だけ移動
                rayX += unitCenter.x;
                rayY += unitCenter.y;
                rayZ += unitCenter.z;
                count++;
                // cout << "ray :" << rayX << ", " << rayY << ", " << rayZ << endl;
            }

            // 3次元画像内に入らなかった場合，　　ループを抜ける　
            if (count > diameter){continue;}

            // 3次元画像内での処理˜
            // cout << "ray :" << rayX << ", " << rayY << ", " << rayZ << endl;
            while((rayX > 0 && rayX < (2 * center.x)) && (rayY > 0 && rayY < (2 * center.y)) && (rayZ > 0 && rayZ < (2 * center.z))){
                // cout << "ray :" << rayX << ", " << rayY << ", " << rayZ << endl;
                // 3次元画像を参照する際の座標
                // 一番近い値を取得する
                int bufferX = static_cast<int>(rayX + 0.5 - center.x);
                int bufferY = static_cast<int>(rayY + 0.5 - center.y);
                int bufferZ = static_cast<int>(rayZ + 0.5 - center.z);

                // 画像幅を超えた場合の処理
                if(bufferX >= width) {bufferX--;}
                if(bufferY >= height) {bufferY--;}
                if(bufferZ >= depth) {bufferZ--;}

                // 3次元画像におけるvoxel値を取得
                unsigned char voxelValue = ucharBuffer[bufferX + bufferY * width + width * height * bufferZ];

                // 最大値を更新したか確認
                if (max < voxelValue){
                    max = voxelValue;
                }

                // 光線を単位ベクトル分だけ移動
                rayX += unitCenter.x;
                rayY += unitCenter.y;
                rayZ += unitCenter.z;
            }
            // cout << "ray :" << rayX << ", " << rayY << ", " << rayZ << endl;

            // Mip imageに最大値を格納
            mipImage[x + y * diameter] = max;
        } 

        // x方向への進みをリセット
        mipX -= horizotal.x;
        mipY -= horizotal.y;
        mipZ -= horizotal.z;
    }
}


int main(){
    string parameterPath = "./data/ProcessingParameter.txt";
    string mhdPath = "./data/output.mhd";
    string rawPath = "./data/output.raw";
    // string mhdPath = "./data/ChestCT/ChestCT.mhd";
    // string rawPath = "./data/ChestCT/ChestCT.raw";
    string saveMhd = "./data/test.mhd";
    string saveRaw = "./data/test.raw";

    Mip mip;
    mip.loadParameter(parameterPath);
    mip.loadMhd(mhdPath);
    mip.loadRaw(rawPath);
    mip.windowLeveling();
    // mip.calcRadius();
    // vector<unsigned char>buffer = mip.getUnsingedBuffer();
    // vector<short>buffer = mip.getShortBuffer();
    mip.MipCreate();
    vector<unsigned char>mipImage = mip.getMip();
    cout << "Mip size : " << mipImage.size() << endl;
    // mip.saveRaw(buffer, saveRaw);
    mip.saveRaw(mipImage, saveRaw);
    mip.saveMhd(saveMhd);
}