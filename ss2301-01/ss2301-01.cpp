#include <iostream>
#include <string>
#include <typeinfo>
#include <type_traits>


int charCount = 0;  // 引数の中に含まれている文字をカウント


template<typename T>
double toDouble(T value){
    return static_cast<double>(value);
}

template<typename T>
T minimum(T arg){
    if((std::is_same<T, std::string>::value) || 
        (std::is_same<T, char>::value) ||
        (std::is_same<T, const char*>::value)){
            charCount++;
            std::cout << "文字が" << charCount << "個含まれています." << std::endl;
    }
    return arg;
}


// Todo: 引数に文字列が含まれている場合の処理を実装する．
template<typename T, typename... Rest>
double minimum(T arg, Rest... rest){

    // 文字が含まれている場合，そのことを出力させる．
    if((std::is_same<T, std::string>::value) || 
        (std::is_same<T, char>::value) ||
        (std::is_same<T, const char*>::value)){
            charCount++;
            std::cout << "文字が" << charCount << "個含まれています." << std::endl;
    }
    double firstValue = toDouble(arg);
    double minValue = minimum(rest...);
    return (firstValue > minValue ? minValue : firstValue);
}



int main(){
    auto result1 = minimum(1,2,3);
    auto result2 = minimum(5.1, 4.5, 2.9);
    auto result3 = minimum(1, 1.1, 2, 2.5, 0.9);
    std::cout << "int: " << result1 << std::endl;
    std::cout << "double: " << result2 << std::endl;
    std::cout << "int and double: " << result3 << std::endl;

    // 文字が含まれている場合 
    auto result4 = minimum(1, 2, 'c');
    std::cout << "char: " << result4 << std::endl;
    
    return 0;
}