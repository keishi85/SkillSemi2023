
//
// Skill seminer05
// �����F�͖�R�M�q�@�w���ԍ��F18T0811M �ŏI�X�V���F2020/10/28
// �g�p�G�f�B�^�FVisual Studio
//

//���s������
// 1 �e�L�X�g�t�@�C���̃t�@�C���p�X
// 2 �C���[�W�t�@�C���̃t�@�C���p�X
// 3 �𑜓x


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


#define MIN16bit -32768 //8�r�b�g�̍ő�l
#define MAX16bit 32767   //8�r�b�g�̍ŏ��l
#define ARGUMENT 4
#define MAX8bit 255 //8�r�b�g�̍ő�l
#define MIN8bit 0   //8�r�b�g�̍ŏ��l


template <class T>
class Image3d {
public:


    T* image3d;          //��f�l���i�[���铮�I�z��

    int _Width3d;          //�摜��x����
    int _Height3d;         //�摜��y����
    int _Depth3d;          //�摜��z����
    float _resoX3d;       //x�����̉𑜓x
    float _resoY3d;       //y�����̉𑜓x
    float _resoZ3d;       //z�����̉𑜓x
    int _WindowLevel3d;    //�E�B���h�E���x��
    int _WindowWidth3d;    //�E�B���h�E��



    //�R���X�g���N�^
    Image3d(int width, int height, int depth, float resox, float resoy, float resoz, int windowlevel, int windowwidth)
        :_Width3d(width), _Height3d(height), _Depth3d(depth), _resoX3d(resox), _resoY3d(resoy), _resoZ3d(resoz), _WindowLevel3d(windowlevel), _WindowWidth3d(windowwidth)
    {
        image3d = new T[width * height * depth];
    }

    //�ڏ��R���X�g���N�^    
    Image3d(int width, int height, int depth, float resox, float resoy, float resoz)
        :Image3d(width, height, depth, resox, resoy, resoz, 0, 0) {}

    ~Image3d() {
        //�f�X�g���N�^
        delete[] image3d;
    }

};

template<class T>
class Image2d {
public:

    T* image2d;          //��f�l���i�[���铮�I�z��

    int _Width2d;          //�摜�̕�
    int _Height2d;         //�摜�̍���
    int _WindowLevel2d;    //�E�B���h�E���x��
    int _WindowWidth2d;    //�E�B���h�E��

    //�R���X�g���N�^
    Image2d(int width, int height, int windowlevel, int windowwidth) :_Width2d(width), _Height2d(height), _WindowLevel2d(windowlevel), _WindowWidth2d(windowwidth)
    {
        image2d = new T[width * height];
    }

    //�ڏ��R���X�g���N�^    
    Image2d(int width, int height)
        :Image2d(width, height, 0, 0) {}


    ~Image2d() {
        //�f�X�g���N�^
        delete[] image2d;
    }

};

class Coordinate3d {
public:
    float X;
    float Y;
    float Z;

    //�R���X�g���N�^
    Coordinate3d(float corX, float corY, float corZ) :X(corX), Y(corY), Z(corZ) {}
    //�ڏ��R���X�g���N�^ 
    Coordinate3d() :Coordinate3d(0, 0, 0) {}

};

class Vector3d {
public:
    float X;        //X���W
    float Y;        //Y���W
    float Z;        //Z���W

    //�R���X�g���N�^
    Vector3d(float vecX, float vecY, float vecZ) :X(vecX), Y(vecY), Z(vecZ) {}
    //�ڏ��R���X�g���N�^�@�@
    Vector3d() :Vector3d(0, 0, 0) {}

};

template <class T>
void ShowImage(string Imgfilename, Image2d<T>& img) {

    /*ShowImage�֐�
    *�@�����F�t�@�C���ɉ摜�f�[�^���o�͂���֐�
    *�@�����F�t�@�C����filename,Image2d�N���X�I�u�W�F�N�gimg
    *�@�߂�l�F�Ȃ�
    */

    ofstream ofsImg(Imgfilename, ios::binary);
    if (!ofsImg) {
        //�t�@�C�����J���Ȃ������ꍇ
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
    /*GetData�֐�
    *�@�����F�t�@�C������C�摜�̕��C�����C�E�B���h�E���x���C�E�B���h�E���C�f�[�^�^�C�v��ǂݍ��ފ֐�
    *�@�����F�t�@�C���p�X���C�摜�̕��C�����C�E�B���h�E���x���C�E�B���h�E���C�f�[�^�^�C�v������ϐ��̎Q��
    *�@�߂�l�F�Ȃ�
    */

    ifstream ifs(path);
    string str;

    if (ifs.fail()) {
        //�t�@�C�����J���Ȃ������ꍇ
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
    /*GradationProcessing�֐�
    *�@�����F�K���������s���֐�
    *�@�����FImage2d�N���X�I�u�W�F�N�gOrgimg,Graimg
    *�@�߂�l�F�Ȃ�
    */

    int min = Orgimg._WindowLevel2d - (Orgimg._WindowWidth2d / 2);    //�K���O�̍ő��f�l
    int max = Orgimg._WindowLevel2d + (Orgimg._WindowWidth2d / 2);    //�K���O�̍ŏ���f�l

    int i = 0;  //�p�����[�^

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

    /*GetImage�֐�
    *�@�����F�t�@�C������摜��ǂݍ��ފ֐�
    *�@�����F�t�@�C���p�X��path,Image3d�N���X�I�u�W�F�N�gimg
    *�@�߂�l�F�Ȃ�
    */

    ifstream ifs(path, ios::binary);
    if (!ifs) {
        //�t�@�C�����J���Ȃ������ꍇ
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

    /*Rotation�֐�
    *�@�����F���W�𒆐S�ʒu���w�肵�ĉ�]������֐�
    *�@�����F��]������p�x�ł̉�]�s��rotationArray,���̍��W�C�ړ���̍��W�C��]���S�̍��W������Coorinate3d�I�u�W�F�N�g
    *�@�߂�l�F�Ȃ�
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

    /*Mip�֐�
    *�@�����F�R�����摜����Mip�摜���o�͂���֐�
    *�@�����F�O�����摜�I�u�W�F�N�gimg,Mip�摜������I�u�W�F�N�gGetImg,�𑜓xresolution
    *�@�߂�l�F�Ȃ�
    */

    //�O�����摜�̒��S���W�������Coordinate3d�I�u�W�F�N�g    
    Coordinate3d center((img._Width3d * img._resoX3d) / 2, (img._Height3d * img._resoY3d) / 2, (img._Depth3d * img._resoZ3d) / 2);

    float R = sqrt(pow(center.X, 2) + pow(center.Y, 2) + pow(center.Z, 2));   //���a�̎Z�o
    float diameter = R * 2; //���a�̎Z�o

    //��]�O��Mip�摜�̒��S���W�������Coordinate3d�I�u�W�F�N�g
    Coordinate3d Sqrtcenter(center.X, center.Y - R, center.Z);
    //��]�O��Mip�摜��3�̒��_�̍��W�������Coordinate3d�I�u�W�F�N�g 
    Coordinate3d Sqrtvertex1(center.X - R, center.Y - R, center.Z - R);
    Coordinate3d Sqrtvertex2(center.X - R, center.Y - R, center.Z + R);
    Coordinate3d Sqrtvertex3(center.X + R, center.Y - R, center.Z - R);


    //�p�x�����W�A���ɕϊ�����D
    float rad = M_PI / 180;
    float radX, radY, radZ;
    radX = rad * (float)rotAngleX;
    radY = rad * (float)rotAngleY;
    radZ = rad * (float)rotAngleZ;

    //��]�̂��߂ɃT�C���R�T�C���̌v�Z���s��
    float SinX, SinY, SinZ, CosX, CosY, CosZ;
    SinX = sin(radX);
    SinY = sin(radY);
    SinZ = sin(radZ);
    CosX = cos(radX);
    CosY = cos(radY);
    CosZ = cos(radZ);

    //��]�s����i�[���邽�߂̔z��
    float roundArray[9] = { CosZ * CosY, CosZ * SinY * SinX + SinZ * CosX, SinZ * SinX - CosZ * SinY * CosX,
                            -SinZ * CosY, CosZ * CosX - SinZ * SinY * SinX, SinZ * SinY * CosX + CosZ * SinX,
                            SinY,       -CosY * SinX,                      CosY * CosX };

    //��]���Mip�摜�̒��S���W�������Coordinate3d�I�u�W�F�N�g
    Coordinate3d rotSqrtcenter;
    Rotation(roundArray, Sqrtcenter, rotSqrtcenter, center);

    //��]���Mip�摜��3�̒��_�̍��W�������Coordinate3d�I�u�W�F�N�g
    Coordinate3d rotSqrtvertex1;
    Rotation(roundArray, Sqrtvertex1, rotSqrtvertex1, center);

    Coordinate3d rotSqrtvertex2;
    Rotation(roundArray, Sqrtvertex2, rotSqrtvertex2, center);

    Coordinate3d rotSqrtvertex3;
    Rotation(roundArray, Sqrtvertex3, rotSqrtvertex3, center);

    //���S�Ɍ������P�ʃx�N�g���i�𑜓x���j
    Vector3d unitVector((center.X - rotSqrtcenter.X) / R,
        (center.Y - rotSqrtcenter.Y) / R,
        (center.Z - rotSqrtcenter.Z) / R);

    //Mip�摜�̏c���̃x�N�g��
    //�c
    Vector3d Vertical(rotSqrtvertex2.X - rotSqrtvertex1.X,
        rotSqrtvertex2.Y - rotSqrtvertex1.Y,
        rotSqrtvertex2.Z - rotSqrtvertex1.Z);
    //��
    Vector3d Horizonal(rotSqrtvertex3.X - rotSqrtvertex1.X,
        rotSqrtvertex3.Y - rotSqrtvertex1.Y,
        rotSqrtvertex3.Z - rotSqrtvertex1.Z);

    //Mip�摜�̏c���̒P�ʃx�N�g��(�𑜓x��)
    //�c
    Vector3d unitVertical(Vertical.X / resolution,
        Vertical.Y / resolution,
        Vertical.Z / resolution);
    //��
    Vector3d unitHorizonal(Horizonal.X / resolution,
        Horizonal.Y / resolution,
        Horizonal.Z / resolution);

    //Mip�摜��𓮂��p�����[�^
    float paramX = rotSqrtvertex1.X,
        paramY = rotSqrtvertex1.Y,
        paramZ = rotSqrtvertex1.Z;

    //�����������̍��W
    float moveX, moveY, moveZ;

    //3�����摜�̕���
    int planeImg = img._Width3d * img._Height3d;


    T max;  //�ő�l���i�[����ϐ�


    int p;  //�O�ډ~�̒[�܂ōs�������𔻒f���邽�߂̃p�����[�^

    int _paramImg, _paramX, _paramY, _paramZ; //�摜�̈ʒu�����邽�߂̃p�����[�^


    for (int Height2d = 0; Height2d < resolution; Height2d++, paramX += unitVertical.X, paramY += unitVertical.Y, paramZ += unitVertical.Z) {

 

            for (int Width2d = 0; Width2d < resolution; Width2d++, paramX += unitHorizonal.X, paramY += unitHorizonal.Y, paramZ += unitHorizonal.Z) {
                    //�����������̏����ʒu�̊i�[
                    moveX = paramX, moveY = paramY, moveZ = paramZ;

                //�ő�l��short�^�̍ŏ��l�ɂ���
                max = MIN16bit;

                //�O�ډ~�̒[�܂ōs�������𔻒f���邽�߂̃p�����[�^
                p = 0;

                //3d�摜�l�ɓ���܂ł̏����i����Ȃ������ꍇ��p�Ŕ��f����j
                while ((0 > moveX || moveX > (2 * center.X) || 0 > moveY || moveY > (2 * center.Y) || 0 > moveZ || moveZ > (2 * center.Z)) && p < diameter) {
                    //�P�ʃx�N�g�����̌����̈ړ�
                    moveX += unitVector.X;
                    moveY += unitVector.Y;
                    moveZ += unitVector.Z;
                    p++;

                }
                //����Ȃ������ꍇ��for���[�v�𔲂���
                if (p == diameter) {
                    continue;
                }

                //3d�摜�ɓ������ꍇ�̏����i�ł��甲����j
                while ((0 <= moveX && moveX < (2 * center.X)) && (0 <= moveY && moveY < (2 * center.Y)) && (0 <= moveZ && moveZ < (2 * center.Z))) {


                    _paramX = (int)(moveX / img._resoX3d + 0.5);
                    _paramY = (int)(moveY / img._resoY3d + 0.5);
                    _paramZ = (int)(moveZ / img._resoZ3d + 0.5);

                    //�摜�̕��𒴂������ԋ߂��l�Œu�������� 
                    if (_paramX >= img._Width3d) {
                        _paramX = img._Width3d - 1;
                    }
                    if (_paramY >= img._Height3d) {
                        _paramY = img._Height3d - 1;
                    }
                    if (_paramZ >= img._Depth3d) {
                        _paramZ = img._Depth3d - 1;
                    }

                    //�����ʒu�̉摜�̍��W�̌v�Z
                    _paramImg = _paramZ * planeImg + _paramY * img._Width3d + _paramX;

                    //�ő�l���X�V������u��������
                    if (max < img.image3d[_paramImg]) {
                        max = img.image3d[_paramImg];
                    }

                    //�P�ʃx�N�g�����̌����̈ړ�
                    moveX += unitVector.X;
                    moveY += unitVector.Y;
                    moveZ += unitVector.Z;
                }


                //Mip�摜�ɍő�l���i�[
                GetImg.image2d[Height2d * resolution + Width2d] = max;

            }


        //Mip�摜�̃p�����[�^��߂��D
        paramX -= Horizonal.X;
        paramY -= Horizonal.Y;
        paramZ -= Horizonal.Z;

    }

}

bool check_int(string str)
{
    /*
        check_int�֐�
        �����F�擾���������񂪐������𔻒f����֐�
        �����Fstring�^�ϐ�
        �߂�l�FTrue or False

    */
    if (all_of(str.cbegin(), str.cend(), isdigit))
    {
        return true;
    }
        cout << "int�^�����ł�����x���͂��Ă�������" << endl;
    return false;
}

int main(int argc, char* argv[])
{
    //�����̌�����v���Ȃ������ꍇ�͏I��
    if (argc != ARGUMENT) {
        cout << "Argument is not appropriate" << endl;
        return -1;
    }

    //�p�X�����΃p�X�ɕϊ�
    fs::path abs_p1{ argv[1] };
    fs::path rel_p1 = fs::relative(abs_p1);

    //�t�@�C�������݂��Ȃ�������I��
    if (!fs::exists(rel_p1)) {
        std::cout << ".....No txtFile exist....." << std::endl;
        return -1;
    }

    //�f�[�^������ϐ��̒�`
    int width, height, depth, windowlevel, windowwidth;
    float resox, resoy, resoz;

    //�f�[�^���e�L�X�g�t�@�C�����璊�o����
    GetData(rel_p1, width, height, depth, resox, resoy, resoz, windowlevel, windowwidth);

    //�p�X�����΃p�X�ɕϊ�
    fs::path abs_p2{ argv[2] };
    fs::path rel_p2 = fs::relative(abs_p2);

    //�t�@�C�������݂��Ȃ�������I��
    if (!fs::exists(rel_p2)) {
        std::cout << ".....No imgFile exist....." << std::endl;
        return -1;
    }

    //�摜������Image3d�I�u�W�F�N�g�̍쐬
    Image3d<short> OrgImg(width, height, depth, resox, resoy, resoz, windowlevel, windowwidth);

    //�摜�f�[�^���I�u�W�F�N�g�Ɋi�[����
    GetImage(rel_p2, OrgImg);

    //���s����������𑜓x�̎擾
    int resolution = atoi(argv[3]);

    //Mip�摜������Image2d�I�u�W�F�N�g�̍쐬
    Image2d<short> MipImg(resolution, resolution, OrgImg._WindowLevel3d, OrgImg._WindowWidth3d);

    string strAngle;
    int rotationangleX, rotationangleY, rotationangleZ;

    do{

        cout << "���ꂼ��̕����̉�]�p�x����(�p�x�œ��͂��ĉ�����)\n���ʂ������ʒu�ł�\nx:";
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

    //Mip����
    Mip(OrgImg, MipImg, resolution,rotationangleX,rotationangleY,rotationangleZ);

    //�~��������̉摜������Image2d�I�u�W�F�N�g�̍쐬
    Image2d<unsigned char> MipGraImg(MipImg._Width2d, MipImg._Height2d);

    //�~���������s��
    GradationProcessing(MipImg, MipGraImg);

    //Mip�摜�̏o��
    ShowImage("mip.raw", MipGraImg);

}