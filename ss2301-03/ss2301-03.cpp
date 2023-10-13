#include <iostream>
#include <string>
#include <fstream>

class FileAnalysis{
    std::string fileName;
public:
    FileAnalysis() : fileName("nothing"){};
    FileAnalysis(std::string fileName) : fileName(fileName){};
    bool isBinaryFile();
    void showJudgement();
};

bool FileAnalysis::isBinaryFile(){
    std::ifstream file(fileName, std::ios::binary);
    if(!file.is_open()){
        std::cerr << "cannot open file" << std::endl;
        return false;
    }

    char c;
    while(file.get(c)){
        // 制御文字の範囲を特定
        if(c < 0x09 || (c > 0x0D && c < 0x20)){
            return true;
        }
        else{
            return false;
        }
    }
    return true;
}

void FileAnalysis::showJudgement(){
    if(FileAnalysis::isBinaryFile()){
        std::cout << "[BIN]" << fileName << std::endl;
    }
    else{
        std::cout << "[TXT]" << fileName << std::endl;
    }
}



int main(){
    // テキストファイル
    FileAnalysis file("t.txt");
    file.showJudgement();

    // バイナリファイル
    FileAnalysis file2("img.png");
    file2.showJudgement();
    return 0;
}