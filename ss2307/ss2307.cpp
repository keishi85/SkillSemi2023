#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <limits>
#include <any>
#include <cmath>

// mhdファイル情報の取得
class MhdInformation {
    protected:
        // mhdファイルの各項目の変数を保持
        std::string object_type;
        std::string n_dims;
        int dim_size_x;
        int dim_size_y;
        int dim_size_z;
        std::string element_type;
        double element_spacing_x;
        double element_spacing_y;
        double element_spacing_z;
        std::string element_byte_order_msb;
        std::string element_data_file;
};

// 画素値の距離
class DistanceInformation{
    protected:
        double distance_x[4];
        double distance_y[4];
        double distance_z[4];
};

class Image : public MhdInformation, public DistanceInformation{
    
    public:
        // mhdファイル操作関数
        std::string get_mhd_information(std::string& mhd_file_name) {
            
            // 拡張子がない場合の追加
            if (mhd_file_name.find(".") == std::string::npos){
                mhd_file_name += ".mhd";
            }
            // mhdファイルからの読み込みができない時
            std::ifstream file(mhd_file_name);
            if (!file.is_open()) {    
                std::cerr << "Error: File does not exist or unable to open file - " << mhd_file_name << std::endl;
                std::exit(1);
            }

            // 内容の読み込み
            std::string line;
            std::string name;
            while(std::getline(file, line)){
                std::istringstream iss(line);
                iss >> name; // "変数名"
                iss.ignore(std::numeric_limits<std::streamsize>::max(), '='); // '='をスキップ
                if (name == "ObjectType"){
                    iss >> object_type;
                }
                else if (name == "NDims"){
                    iss >> n_dims;
                }
                else if (name == "DimSize"){
                    iss >> dim_size_x >> dim_size_y >> dim_size_z;
                }
                else if (name == "ElementType"){
                    iss >> element_type;
                }
                else if (name == "ElementSpacing"){
                    iss >> element_spacing_x >> element_spacing_y >> element_spacing_z;
                }
                else if (name == "ElementByteOrderMSB"){
                    iss >> element_byte_order_msb;
                }
                else if (name == "ElementDataFile"){
                    iss >> element_data_file;
                }
                else{
                    std::cerr << "Error: Can't save mhd information to variable." << std::endl;
                    std::exit(1);
                }
            }         
            // ファイルを閉じる
            file.close();
            return element_data_file;
        }


        // 結果をrawファイルで出力する関数
        void write_raw_file(const std::string& output_raw_file, std::vector<std::vector<std::vector<short>>>& result_data, const int output_size_x, const int output_size_y, const int output_size_z) {
            std::ofstream file(output_raw_file, std::ios::binary);
            if (!file) {
                std::cerr << "Error: Failed to open the file.\n";
                std::exit(1);
            }
            // 結果の書き出し
            for (int x = 0; x < output_size_x; ++x) {
                for (int y = 0; y < output_size_y; ++y) {
                    file.write(reinterpret_cast<char*>(result_data[y][x].data()), output_size_z*sizeof(short));
                }
            }
            file.close();
        }


        // 結果をmhdファイルで出力する関数
        void write_mhd_file(std::string& output_mhd_file, std::string& output_raw_file, int width, int height, int slice) {
            std::ofstream file(output_mhd_file);
            if (!file) {
                std::cerr << "Error: Failed to open the file.\n";
                std::exit(1);
            }
            // 結果の書き出し
            file << "ObjectType = Image\n";
            file << "NDims = " << n_dims << "\n";
            file << "DimSize = " << width << " " << height << " " << slice << "\n";
            file << "ElementType = MET_SHORT\n";
            file << "ElementSpacing = "<< element_spacing_x << " " << element_spacing_y << " " << element_spacing_z << "\n";
            file << "ElementByteOrderMSB = False\n";
            file << "ElementDataFile = " << output_raw_file << "\n";

            file.close();
        }



        // ターゲットの画素値との距離を計算する関数(クラスに保存)
        void calculate_distance(double target_x, double target_y, double target_z, int rounding_down_x, int rounding_down_y, int rounding_down_z){
                distance_x[0] = 1 + target_x - rounding_down_x;
                distance_x[1] = target_x - rounding_down_x;
                distance_x[2] = rounding_down_x + 1 - target_x;
                distance_x[3] = rounding_down_x + 2 - target_x;
                distance_y[0] = 1 + target_y - rounding_down_y;
                distance_y[1] = target_y - rounding_down_y;
                distance_y[2] = rounding_down_y + 1 - target_y;
                distance_y[3] = rounding_down_y + 2 - target_y;
                distance_z[0] = 1 + target_z - rounding_down_z;
                distance_z[1] = target_z - rounding_down_z;
                distance_z[2] = rounding_down_z + 1 - target_z;
                distance_z[3] = rounding_down_z + 2 - target_z;
        }

        // h(t)の計算
        double calculate_weight(double t){
            int a = -1;
            double h;
            double abs_number = std::abs(t);

            if (abs_number <= 1){
                h = (a + 2)*abs_number*abs_number*abs_number - (a + 3)*abs_number*abs_number +1;
            }
            else if (abs_number > 2){
                h = 0;
            }
            else{
                h = a*abs_number*abs_number*abs_number -5*a*abs_number*abs_number + 8*a*abs_number - 4*a;
            }
            return h;
        }

        // 補間する画素値の計算
        double calculate_interpolation(double target_x, double target_y, double target_z, std::vector<std::vector<std::vector<short>>>& image){
            double interpolation = 0.0;
            // 重み変数
            double h_x;
            double h_y;
            double h_z;

            
            // 補間するtargetの画素値の小数点以下を切り捨て
            int rounding_down_x = floor(target_x);
            int rounding_down_y = floor(target_y);
            int rounding_down_z = floor(target_z);

            // 距離の計算
            calculate_distance(target_x, target_y, target_z, rounding_down_x, rounding_down_y, rounding_down_z);
            // std::cout << "Target" << rounding_down_x << std::endl;

            // 補間する画素値の範囲         
            for (int y = rounding_down_y - 1; y <= rounding_down_y + 2; y++){
                for (int x = rounding_down_x - 1; x <= rounding_down_x + 2 ; x++){
                    for (int z = rounding_down_z - 1; z <= rounding_down_z + 2 ; z++){
                        // 参照する画素値がない時
                        if (x<0 || y<0 || z<0 || x >= dim_size_x || y >= dim_size_y || z >= dim_size_z ){
                            interpolation += 0;
                        }
                        else{
                            h_x = calculate_weight(distance_x[x - (rounding_down_x-1)]);
                            h_y = calculate_weight(distance_y[y - (rounding_down_y-1)]);
                            h_z = calculate_weight(distance_z[z - (rounding_down_z-1)]);
                            interpolation += (image[y][x][z] * h_x * h_y * h_z);
                        }
                        // std::cout << "z" << z << std::endl;
                    }
                    // std::cout << "x" << x << std::endl;
                }
                // std::cout << "y" << y << std::endl;
            }
            return interpolation;
        }

     
        // 画像を読み込み各種操作を指示する関数
        void get_image_information(std::string&element_data_file, std::string&output_file) { 
            
            // ".raw"の拡張子がついていない時，拡張子を付け足す
            if (element_data_file.find(".") == std::string::npos){
                element_data_file += ".raw";
            }  

            // raw画像をバイナリモードで開く
            std::ifstream raw_file(element_data_file, std::ios::binary);
            if (!raw_file) {
                std::cerr << "Error: Unable to open rawfile - " << std::endl;
                std::exit(1);  
            }

            // raw画像データ(バイナリーモード)をベクトルに読み込む
            std::vector<std::vector<std::vector<short>>> image(dim_size_y, std::vector<std::vector<short>>(dim_size_x, std::vector<short>(dim_size_z)));
            image.resize(dim_size_y, std::vector<std::vector<short>>(dim_size_x, std::vector<short>(dim_size_z)));
            
            for (int x = 0; x < dim_size_x; ++x) {
                for (int y = 0; y < dim_size_y; ++y) {
                raw_file.read(reinterpret_cast<char*>(image[y][x].data()), dim_size_z*sizeof(short));
                }
            }

            // 補間処理プログラム
            // 出力画像の配列準備
            int output_size_x = dim_size_x;
            int output_size_y = dim_size_y;
            int output_size_z = static_cast<int>(std::floor(static_cast<double>(dim_size_z) * element_spacing_z / element_spacing_x));
            std::vector <std::vector <std::vector <short>>> result_data;
            result_data.resize(output_size_y, std::vector<std::vector<short>>(output_size_x, std::vector<short>(output_size_z)));
            // 解像度の変更
            element_spacing_z = element_spacing_x;

            // 補間実施
            for (int z = 0; z < output_size_z; ++z) {
                for (int x = 0; x < output_size_x; ++x) {
                    for (int y = 0; y < output_size_y; ++y) {
                        double target_x = static_cast<double>((x) * (dim_size_x - 1) / (output_size_x - 1));
                        double target_y = static_cast<double>((y) * (dim_size_y - 1) / (output_size_y - 1));
                        double target_z = static_cast<double>((z) * (dim_size_z - 1) / (output_size_z - 1));

                        result_data[y][x][z] = calculate_interpolation(target_x, target_y, target_z, image);
                    
                        // std::cout << "z" << z << std::endl;
                    }
                    // std::cout << "x" << x << std::endl;
                }
                std::cout << z << std::endl;    
            }   

            // mhdファイル，rawファイルへフィルタ後の結果を出力するための準備
            std::string output_mhd_file;
            std::string output_raw_file;
            // ".mhd"，".raw"の拡張子がついていない時，拡張子を付け足す
            if (output_file.find(".") == std::string::npos){
                output_mhd_file = output_file + ".mhd";
                output_raw_file = output_file + ".raw";
            }
            else{
                std::cerr << "Error: Don't add extnsion to output file name- " << std::endl;
                std::exit(1);                 
            }
                       
            // // rawファイル，mhdファイルへの書き込み
            write_raw_file(output_raw_file, result_data, output_size_x, output_size_y, output_size_z);
            write_mhd_file(output_mhd_file, output_raw_file, output_size_x, output_size_y, output_size_z);
        } 
};


int main (int argc, char *argv[]){

    // コマンド引数の条件に反した時
    if (argc!=3){
        std::cerr<<"Enter the command like this:" << argv[0] << "<input_filename>.mhd" << "<out_filename>" << std::endl;
        return 1;
    }

    // mhdファイル名取得
    std::string input_mhd_file = argv[1];
    // 出力ファイル名取得
    std::string output_file = argv[2];
    
    // 変数宣言
    std::string input_raw_file;
    Image ct;

    // mhdファイルから情報取得
    input_raw_file = ct.get_mhd_information(input_mhd_file);

    // mhdファイルの情報からrawファイルの読み込みと画像の補間
    ct.get_image_information(input_raw_file, output_file);

    return 0;
}