#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

std::vector<std::string> articlesList = {"the", "a", "an"};
std::vector<std::string> conjuctionList = {
    "and", "but", "or", "nor", "for", "yet"
};
std::vector<std::string> prepositionList = {
    "in", "on", "at", "by", "with", "of", "to", "from", "for", "during", "after", "before", "between"
};





class TextAnalysis{
    std::string text;
    std::vector<std::string> notCapitalizeWords;
public:
    // コンストラクタではテキストの読み込み, 前置詞，等位接続詞，冠詞をファイルから読み込む．
    TextAnalysis(std::string text) : text(text){}
    bool isArtiConjPrep(std::string word);  // 前置詞，等位接続詞，冠詞の場合，trueを返す.
    bool isIncludedNotCapitalizeWord(std::string word); // 大文字にしない単語をファイルから指定する場合，その判定を行う関数
    int loadTextFile(std::string fileName);
    void capitalize();
    void capitalize(std::string fileName);
    std::string get(){ return this->text; };
    void clear(){ notCapitalizeWords.clear();}; // 大文字にしない単語のデータを消去する
};

bool TextAnalysis::isArtiConjPrep(std::string word){
    bool isArticle = std::find(articlesList.begin(), articlesList.end(), word) != articlesList.end();
    bool isConjuction = std::find(conjuctionList.begin(), conjuctionList.end(), word) != conjuctionList.end();
    bool isPreposition = std::find(prepositionList.begin(), prepositionList.end(), word) != prepositionList.end();
    if(isArticle || isConjuction || isPreposition){
        return true;
    }
    else{
        return false;
    }
}

bool TextAnalysis::isIncludedNotCapitalizeWord(std::string word){
    bool isCheck = std::find(notCapitalizeWords.begin(), notCapitalizeWords.end(), word) != notCapitalizeWords.end();
    if(isCheck){
        return true;
    }
    else{
        return false;
    }
}


int TextAnalysis::loadTextFile(std::string fileName){
    std::fstream file(fileName);
    if(!file){
        std::cerr << " cannot open file." << std::endl;
        return 1;
    }

    std::string line;
    while(std::getline(file, line)){
       std::stringstream iss(line);
       std::string word;
       while(std::getline(iss, word, ',')){
            notCapitalizeWords.push_back(word);
       } 
    }
    return 0;
}



void TextAnalysis::capitalize(){
    std::vector<std::string> tokens;
    std::stringstream iss(this->text);
    std::string token;
    while(iss >> token){
        tokens.push_back(token);
    }    
    for(auto& word : tokens){
        if(!word.empty())
            // 前置詞，等位接続詞，冠詞ではない場合, 先頭の文字を大文字にする． 
            if(!(TextAnalysis::isArtiConjPrep(word))){
                word[0] = std::toupper(word[0]);
            }
    }
    std::stringstream ss;
    for(auto word : tokens){
        ss << word << " ";
        // std::cout << word << std::endl;
    }
    this->text = ss.str();
}

void TextAnalysis::capitalize(std::string fileName){
    TextAnalysis::loadTextFile(fileName);
    std::vector<std::string> tokens;
    std::stringstream iss(this->text);
    std::string token;
    while(iss >> token){
        tokens.push_back(token);
    }    
    for(auto& word : tokens){
        if(!word.empty())
            // 前置詞，等位接続詞，冠詞ではない場合, 先頭の文字を大文字にする． 
            if(!(TextAnalysis::isIncludedNotCapitalizeWord(word))){
                word[0] = std::toupper(word[0]);
            }
    }
    std::stringstream ss;
    for(auto word : tokens){
        ss << word << " ";
        //std::cout << word << std::endl;
    }
    this->text = ss.str();
}



int main(){
    TextAnalysis text("The cat and the dog, for example, played together.");
    text.capitalize();
    std:: cout << text.get() << std::endl;

    // 前置詞，等位接続詞，冠詞をまとめたファイルが notCapitalizedWords.txt
    text.capitalize("notCapitalizedWords.txt");
    std:: cout << text.get() << std::endl;

    return 0;
}
