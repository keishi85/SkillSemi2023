# include <iostream>
# include <fstream>
# include <string>
# include <sstream>
# include <vector>
# include <map>

# include "mhdRaw.hpp"      // To load and write mhd and raw files

using namespace std;


class TricubicInterpolator : LoadMhdRaw{
    map<string, string> mhd;
    int width, height, depth;
    vector<short> buffer;
    std::vector<short> interpolatedData; // Save imputed data

public:
    TricubicInterpolator(string mhdPath, string rawPath);
    vector<short>getBuffer(){return this->buffer;};

    void saveRaw(const vector<short>& buffer, string savePath){saveRawImage(buffer, savePath);};
    void createMhd(string savePath){createMhdFile(savePath);};

    // Implement Tricubic
    double weight(double x);
    short interpolateAt(const std::vector<short>& buffer, double x, double y, double z);
    short getVoxelValue(const std::vector<short>& buffer, int x, int y, int z);
    std::vector<short> tricubicInterpolate(const std::vector<short>& buffer);
};

TricubicInterpolator::TricubicInterpolator(string mhdPath, string rawPath){
    this->mhd = loadmhd(mhdPath);

    if(mhd.find("DimSize") != mhd.end()){
        string dimSize = mhd["DimSize"];
        istringstream iss(dimSize);
        iss >> this->width >> this->height >> this->depth;
    }
    else{
        cout << "DimSizeが定義されていません" ;
        return;
    }

    this->buffer = loadRawImage(rawPath, this->width, this->height, this->depth);
}

// Function to calc weights
double TricubicInterpolator::weight(double x) {
    x = fabs(x);
    if (x <= 1.0) {
        return x * x * (x - 2.0) * 1.5 + 1.0;
    } else if (x < 2.0) {
        return x * (x * (x * -0.5 + 2.5) - 4.0) + 2.0;
    } else {
        return 0.0;
    }
}

// Calculate interpolated value at given position
short TricubicInterpolator::interpolateAt(const std::vector<short>& buffer, double x, double y, double z) {
    double result = 0.0;
    double sumWeights = 0.0;

    for (int dz = -1; dz <= 2; ++dz) {
        for (int dy = -1; dy <= 2; ++dy) {
            for (int dx = -1; dx <= 2; ++dx) {

                int ix = static_cast<int>(std::floor(x)) + dx;
                int iy = static_cast<int>(std::floor(y)) + dy;
                int iz = static_cast<int>(std::floor(z)) + dz;
                
                double voxelValue = static_cast<double>(getVoxelValue(buffer, ix, iy, iz));
                double w = weight(x - ix) * weight(y - iy) * weight(z - iz);
                // std::cout << sumWeights << endl;

                result += voxelValue * w;
                sumWeights += w;
            }
        }
    }
    
    if (sumWeights > 0.0) {
        return static_cast<short>(result / sumWeights);
    }
    return 0;
}

// Helper function to get voxel values from buffer
short TricubicInterpolator::getVoxelValue(const std::vector<short>& buffer, int x, int y, int z) {
    // if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) {
    //     return 0; // Out of bounds returns 0
    // }
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    z = std::max(0, std::min(z, depth - 1));
    return buffer[x + y * width + z * width * height];
}

// Main function that performs tricubic interpolation
std::vector<short> TricubicInterpolator::tricubicInterpolate(const std::vector<short>& buffer) {
    // Get ElementSpacing of x, y, z
    double xResolution, yResolution, zResolution;
    if(mhd.find("ElementSpacing") != mhd.end()){
        string dimSize = mhd["ElementSpacing"];
        istringstream iss(dimSize);
        iss >> xResolution >> yResolution >> zResolution;
    }
    else{
        cout << "ElementSpacingが定義されていません" ;
    }

    // Get the ratio of x/y-axis and z-axis resolution to calculate the new z-axis resolution
    double scaleFactor = zResolution  / xResolution;

    // Calculate new depth (z-axis resolution)
    int newDepth = static_cast<int>(std::ceil(depth * scaleFactor));
    cout << "depth : " << newDepth << endl;

    // Initialize the vector that stores the isotropic image data
    std::vector<short> result(width * height * newDepth);
    cout << "xResolution = " << xResolution << ", zResolution = " << zResolution << ", scaleFactor = " << scaleFactor << endl;
    cout << "width = " << width << ", height = " << height << ", new depth = " << newDepth << endl; 

    // For some reason, the time when z=0 is not completed
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            result[x + y * width] = buffer[x + y * width];
        }
    }
    for (int z = 1; z < newDepth; ++z) {
        double zPos = z / scaleFactor; // Calc new z axis
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Calculate interpolated values by calling interpolateAt function
                double interpolatedValue = interpolateAt(buffer, x + 0.5, y + 0.5, zPos);

                // Store result 
                result[x + y * width + z * width * height] = static_cast<short>(interpolatedValue);
            }
        }
    }
    interpolatedData = result;
    cout << "buffer size : " << buffer.size() << endl;
    cout << "new buffer size : " << result.size() << endl;
    return result;
}



int main(){
    string mhdPath = "./ss2307_data/ChestCT.mhd";
    string rawPath = "./ss2307_data/ChestCT.raw";
    TricubicInterpolator ti(mhdPath, rawPath);

    vector<short>buffer = ti.getBuffer();

    // Tricubic interpolator
    vector<short> interpolatedBuffer = ti.tricubicInterpolate(buffer);

    // Save the data
    string saveMhdPath = "./ss2307_data/output.mhd";
    string saveRawPath = "./ss2307_data/output.raw";
    ti.saveRaw(interpolatedBuffer, saveRawPath);
    ti.createMhd(saveMhdPath);
}