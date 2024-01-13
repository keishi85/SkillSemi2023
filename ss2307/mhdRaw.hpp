# include <iostream>
# include <sstream>
# include <map>
# include <fstream>
# include <string>
# include <vector>


class LoadMhdRaw{
    int width = 0, height = 0, depth = 0;
    std::map<std::string, std::string> dic;
    std::vector<short>buffer;
public:
    std::map<std::string, std::string>loadmhd(std::string path);
    std::vector<short> loadRawImage(std::string path, int width, int height, int depth);

    template<typename U>
    void saveRawImage(const std::vector<U>& buffer, const std::string savePath);
    void createMhdFile(const std::string savePath);
    std::string getFileNameFromPath(const std::string& path);
};

std::map<std::string, std::string>LoadMhdRaw::loadmhd(std::string path){
    std::ifstream inFile(path);

    if (!inFile) {
        std::cerr << "ファイルを開けませんでした: " << path << std::endl;
    }
    
    std::string line;
    while(std::getline(inFile, line)){
        std::istringstream iss(line);
        std::string key, value;
        std::getline(iss, key, '=');
        key.erase(key.find_last_not_of(" ") + 1); // Delete a last space 
        iss >> std::ws; // skip space
        std::getline(iss, value);
        value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), value.end());
        dic[key] = value;
    }
    
    return dic;
}

std::vector<short> LoadMhdRaw::loadRawImage(std::string path, int width, int height, int depth) { 
    this->width = width;
    this->height = height;   
    this->depth = depth;
    // Allocate a buffer to store image data
    std::vector<short> buffer(width * height * depth);

    // Load  raw file
    std::ifstream file(path, std::ios::binary);
    if(!file){
        std::cerr << "Cannot open file: " << path << std::endl; 
    }

    // Write buffer from file
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(short));

    // Error if file is smaller than expected
    if (file.gcount() != (buffer.size() * sizeof(short))){
        std::cerr << "File size does not match the expected size" << std::endl;
    }

    file.close();

    this->buffer = buffer;

    return buffer;
}

template<typename U>
void LoadMhdRaw::saveRawImage(const std::vector<U>& buffer, const std::string savePath){
    if (savePath.substr(savePath.find_last_of(".") + 1) != "raw") {
        std::cout << "拡張子を\'.raw\' にしてください．" << std::endl;
        return;
    } 

    std::ofstream outFile(savePath, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(U));
}

void LoadMhdRaw::createMhdFile(const std::string savePath){
    if (savePath.substr(savePath.find_last_of(".") + 1) != "mhd") {
        std::cout << "拡張子を\'.mhd\' にしてください．" << std::endl;
        return;
    } 

    // If type of buffer is short, data type is "MET_SHORT"
    // Else data type is "MET_UCHAR"
    std::string dataType;
    if (std::is_same<decltype(buffer)::value_type, short>::value) {
        dataType = "MET_SHORT";
    } else {
        dataType = "MET_UCHAR";
    }

    std::ofstream mhdFile(savePath);

    std::string rawFileName = LoadMhdRaw::getFileNameFromPath(savePath) + ".raw";

    if(width == 0 || height == 0){
        std::cout << "width, height が定義されていません. \nLoadRawImageを実行してください." << std::endl;
        return;
    }


    // Adjust z resolution to x, y
    double xResolution, yResolution, zResolution;
    if(dic.find("ElementSpacing") != dic.end()){
        std::string dimSize = dic["ElementSpacing"];
        std::istringstream iss(dimSize);
        iss >> xResolution >> yResolution >> zResolution;
    }

    // Get the ratio of x/y-axis and z-axis resolution to calculate the new z-axis resolution
    double scaleFactor = zResolution  / xResolution;
    int newDepth = static_cast<int>(std::ceil(this->depth * scaleFactor));

    mhdFile << "ObjectType = Image\n";
    mhdFile << "NDims = 3\n";
    mhdFile << "DimSize = " << this->width << " " << this->height << " " << newDepth << "\n";
    mhdFile << "ElementType = " << dataType << "\n";
    mhdFile << "ElementSpacing = " <<  xResolution << " " << yResolution << " " << xResolution << "\n";
    mhdFile << "ElementByteOrderMSB = False\n";
    mhdFile << "ElementDataFile = " << rawFileName << "\n";
}

std::string LoadMhdRaw::getFileNameFromPath(const std::string& path) {
    size_t lastSlashPos = path.find_last_of("/\\");
    std::string filename;

    if (lastSlashPos != std::string::npos) {
        filename = path.substr(lastSlashPos + 1); // ファイル名の抽出
    } 
    else {
        filename = path; // スラッシュが見つからない場合は、path全体がファイル名
    }

    // ファイル名から拡張子を削除
    size_t lastDotPos = filename.find_last_of(".");
    if (lastDotPos != std::string::npos) {
        filename = filename.substr(0, lastDotPos);
    }

    return filename;
}



// int main(){
//     LoadMhdRaw med;
//     std::map<std::string, std::string> mhd = med.loadmhd("./ss2307_data/ChestCT.mhd");
//     for (const auto kv : mhd){
//         std::cout << kv.first <<  ", " << kv.second << std::endl;
//     }

//     // Get width and height
//     int width, height;
//     std::string dimSize = mhd["DimSize"];
//     std::istringstream iss(dimSize);
//     iss >> width >> height;

//     std::vector<short>buffer = med.loadRawImage("./ss2307_data/ChestCT.raw", width, height);
// }