#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
template<typename PixelType>
class Image
{
    using DataType = std::vector<std::vector<PixelType>>;

public:
    Image(){};
    void Read(const std::string& fileName);
private:
    DataType Data;
    std::size_t rows{ 0 };
    std::size_t columns{ 0 };
};
template<typename PixelType>
void Image<PixelType>::Read(const std::string& fileName)
{
    cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR);
    if (!image.data) 
    {
        std::cerr << "No image data \n";
        return;
    }
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image", image);
    cv::waitKey(0);
}
int main()
{
    Image<uchar>image;
    image.Read("C:/Users/Maxim/OneDrive/Escritorio/Septimo semestre/Taller cortometraje/imgs/3.jpg");

	return 0;
}


