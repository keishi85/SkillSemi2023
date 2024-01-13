# include <iostream>
# include <map>
# include <fstream>
# include <sstream>
# include <string>
# include <filesystem>


class ImageProcessor{
    // The all following are all paths
    std::filesystem::path parentDir;
    std::map<std::string, std::string> parameter;
    std::map<std::string, std::string> mhdFile;
    std::string rawFile;

    int width, height;

    std::vector<short> shortBuffer; // 16bit image data
    std::vector<unsigned char> ucharBuffer; // 8bit image data


public:
    ImageProcessor();
    std::string getTextFileFromCommandLine();
    std::map<std::string, std::string> getdic(std::string path);

    int loadRawImage();
    void windowLeveling();

    std::vector<short> getShortBuffer(){return this->shortBuffer;};
    std::vector<unsigned char> getUnsignedBuffer(){return this->ucharBuffer;};

    void saveRawImage(const std::vector<unsigned char>& buffer, const std::string saveDic);
    void saveRawImage(const std::vector<short>& buffer, const std::string saveDic);
    void createMhdFile(const std::string& dataType, const std::string saveDic);

    // Filter method
    template<typename T>
    std::vector<T> filterProcess(const std::vector<T>& buffer);

    template<typename T>
    std::vector<T> sobelFilter(const std::vector<T>& buffer, std::string kind);

    template<typename T>
    std::vector<T> movingAverageFilter(const std::vector<T>& buffer);

    template<typename T>
    std::vector<T> medianFilter(const std::vector<T>& buffer);
};

ImageProcessor::ImageProcessor(){
    std::string path = ImageProcessor::getTextFileFromCommandLine();

    std::filesystem::path fsPath(path);
    this->parentDir = fsPath.parent_path().string();
    this->parameter = this->getdic(path);

    // Get the contents written in the mhd file in dictionary format
    if (this->parameter.find("Input") != this->parameter.end()) {
        std::string input = this->parameter["Input"];
        std::string mhdFile = input + ".mhd";
        std::filesystem::path mhdPath = this->parentDir / mhdFile;
       
        this->mhdFile = this->getdic(mhdPath.string());
    } 
    else {
        std::cerr << "Input key not found in parameters." << std::endl;
    }

    // Get raw file path
    std::filesystem::path rawFile = this->parameter["Input"] + ".raw";
    std::filesystem::path rawPath = this->parentDir / rawFile;
    this->rawFile = rawPath.string();
}

std::map<std::string, std::string>ImageProcessor::getdic(std::string path){
    std::ifstream inFile(path);

    if (!inFile) {
        std::cerr << "ファイルを開けませんでした: " << path << std::endl;
    }
    
    std::map<std::string, std::string> dic;
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

std::string ImageProcessor::getTextFileFromCommandLine() {
    std::string filePath;
    std::cout << "テキストファイルのパスを入力してください: ";
    std::cin >> filePath;

    return filePath;
}

int ImageProcessor::loadRawImage() {
    // Get width and height
    std::string dimSize = this->mhdFile["DimSize"];
    std::istringstream iss(dimSize);
    iss >> this->width >> this->height;
    
    // Allocate a buffer to store image data
    std::vector<short> buffer(width * height);

    // Load  raw file
    std::ifstream file(this->rawFile, std::ios::binary);
    if(!file){
        std::cerr << "Cannot open file: " << this->rawFile << std::endl; 
        return 1;
    }

    // Write buffer from file
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(short));

    // Error if file is smaller than expected
    if (file.gcount() != (buffer.size() * sizeof(short))){
        std::cerr << "File size does not match the expected size" << std::endl;
        return 2;
    }

    file.close();

    this->shortBuffer = buffer;

    return 0;
}

void ImageProcessor::windowLeveling(){
    // Finish if WindowProcessing != True
    if(this->parameter["WindowProcessing"] != "True"){
        std::cout << "No gradation conversion." << std::endl;
        return;
    }

    short windowWidth = std::stoi(this->parameter["WindowWidth"]);
    short windowCenter = std::stoi(this->parameter["WindowLevel"]);

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

void ImageProcessor::saveRawImage(const std::vector<unsigned char>& buffer, const std::string saveDic){
    std::string filename = this->parameter["Output"] + ".raw";
    filename = saveDic + "/" + filename;  

    std::ofstream outFile(filename, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(unsigned char));
}

void ImageProcessor::saveRawImage(const std::vector<short>& buffer, const std::string saveDic){
    std::string filename = this->parameter["Output"] + ".raw";
    filename = saveDic + "/" + filename;


    std::ofstream outFile(filename, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(short));
}

void ImageProcessor::createMhdFile(const std::string& dataType, const std::string saveDic){
    std::string filename = this->parameter["Output"] + ".mhd";
    filename = saveDic + "/" + filename;

    std::string rawFileName = this->parameter["Output"];
    rawFileName += ".raw";

    std::ofstream mhdFile(filename);
    mhdFile << "ObjectType = Image\n";
    mhdFile << "NDims = 2\n";
    mhdFile << "DimSize = " << this->width << " " << this->height << "\n";
    mhdFile << "ElementType = " << dataType << "\n";
    mhdFile << "ElementSpacing = 1.000000 1.000000\n";
    mhdFile << "ElementByteOrderMSB = False\n";
    mhdFile << "ElementDataFile = " << rawFileName << "\n";
}

template<typename T>
std::vector<T> ImageProcessor::filterProcess(const std::vector<T>& buffer){
    std::string filterMethod = this->parameter["ImageProcessing"];
    
    if (filterMethod == "SobelX"){
        return ImageProcessor::sobelFilter(buffer, "x");
    }
    else if (filterMethod == "SobelY"){
        return ImageProcessor::sobelFilter(buffer, "y");
    }
    else if (filterMethod == "MovingAverage"){
        return ImageProcessor::movingAverageFilter(buffer);
    }
    else if (filterMethod == "Median"){
        return ImageProcessor::medianFilter(buffer);
    }
    else{
        std::cout << "Cannot find " << filterMethod << std::endl;
        return buffer;
    }
}

template<typename T>
std::vector<T> ImageProcessor::sobelFilter(const std::vector<T>& buffer, std::string kind){
    int width = this->width;
    int height = this->height;

    // Output buffer
    std::vector<T> outputBuffer(width * height, 0);

    for (int y = 1; y < height - 1; y++){
        for (int x = 1; x < width - 1; x++){
            int g = 0; // Use int for intermediate calculations

            if (kind == "x"){
                g = 
                    -1 * buffer[(y - 1) * width + (x - 1)] +
                     0 * buffer[(y-1) * width + (x)] + 
                     1 * buffer[(y-1) * width + (x + 1)] + 
                    -2 * buffer[(y) * width + (x - 1)] + 
                     0 * buffer[(y) * width + (x)] + 
                     2 * buffer[(y) * width + (x + 1)] + 
                    -1 * buffer[(y + 1) * width + (x - 1)] + 
                     0 * buffer[(y + 1) * width + (x)] + 
                     1 * buffer[(y + 1) * width + (x + 1)];
            }
            else if (kind == "y"){
                g =
                    -1 * buffer[(y - 1) * width + (x - 1)] +
                    -2 * buffer[(y - 1) * width + (x)] +
                    -1 * buffer[(y - 1) * width + (x + 1)] +
                     0 * buffer[(y) * width + (x - 1)] +
                     0 * buffer[(y) * width + (x)] +
                     0 * buffer[(y) * width + (x + 1)] +
                     1 * buffer[(y + 1) * width + (x - 1)] +
                     2 * buffer[(y + 1) * width + (x)] +
                     1 * buffer[(y + 1) * width + (x + 1)];
            }

            // Scale and clip the result
            g = std::max(0, std::min(255, std::abs(g)));

            // Store results in output buffer
            outputBuffer[y * width + x] = static_cast<T>(g);
        }
    }

    return outputBuffer;
}

template<typename T>
std::vector<T> ImageProcessor::movingAverageFilter(const std::vector<T>& buffer){
    int kernel = std::stoi(this->parameter["MovingAverageFilterKernel"]);
    int width = this->width;
    int height = this->height;

    std::vector<T> outputBuffer(width * height, 0);

    for (int y = 1; y < height - 1; y++){
        for (int x = 1; x < width -1; x++){
            int sum = 0;
            for (int ky = -1; ky <= 1; ++ky){
                for(int kx = -1; kx <= 1; kx++){
                    sum += buffer[(y + ky) * width + (x + kx)];
                }
            }
            outputBuffer[y * width + x] = sum / 9;
        }
    }
    return outputBuffer;
}

template<typename T>
std::vector<T> ImageProcessor::medianFilter(const std::vector<T>& buffer){
    int kernel = std::stoi(this->parameter["MedianFilterKernel"]);
    int width = this->width;
    int height = this->height;

    std::vector<T> outputBuffer(width * height, 0);

    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            std::vector<T> neighbors;

            // Gather pixel value in filter
            for (int ky = -kernel; ky <= kernel; ky++){
                for (int kx = -kernel; kx <= kernel; kx++){
                    int posY = y + ky;
                    int posX = x +kx;

                    // Apply zero padding to ignore out of image
                    if(posY >= 0 && posY < height && posX >=0 && posX < width){
                        neighbors.push_back(buffer[posY * width + posX]);
                    }
                }
            }
             // Finding median
             std::nth_element(neighbors.begin(), neighbors.begin() + neighbors.size() / 2, neighbors.end());
             T median = neighbors[neighbors.size() / 2];

             outputBuffer[y * width + x] = median;
        }
    }
    return outputBuffer;
}


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
    // buffer = processor.filterProcess(buffer);
    
    std::string savePath = "./data";
    processor.saveRawImage(buffer, savePath);
    processor.createMhdFile(dataType, savePath);
}