#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "RGB.h"
template<typename PixelType>
class Image
{
    using DataType = std::vector<std::vector<PixelType>>;

public:
    Image(){};
    void Read(const std::string& fileName);
private:
    DataType Data;
    std::size_t Rows{ 0 };
    std::size_t Columns{ 0 };
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
    Rows = image.rows;
    Columns = image.cols;

    Data = DataType(Rows, std::vector<PixelType>(Columns, PixelType{} ) );
    uchar red, green, blue;
    for (unsigned r = 0; r < Rows; r++)
    {
        cv::Vec3b * row = image.ptr<cv::Vec3b>(r);
        for (unsigned c = 0; c < Columns; c++)
        {
            red = row[c][2];
            green = row[c][1];
            blue = row[c][0];
            //Poner los valores en nuestra matriz
            if constexpr (std::is_fundamental<PixelType>::value)//char, short, float
            {
                Data[r][c] = static_cast<PixelType>((red + green + blue) / 3);
            }
            else
            {
                Data[r][c][0] = static_cast<typename PixelType::ValueType>(red); 
                Data[r][c][1] = static_cast<typename PixelType::ValueType>(green);
                Data[r][c][2] = static_cast<typename PixelType::ValueType>(blue);
            }
        
        }
    }

    std::cout << image.ptr<cv::Vec3b>(0)[0] << "\n";
    std::cout << +Data[0][0][0]<<","<< +Data[0][0][1]<<","<< +Data[0][0][2]<<"\n";

    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image", static_cast<typename cv::Mat>(Data));
    cv::waitKey(0);
}
int main()
{
    Image<RGB<uchar>>image;
    image.Read("C:/Users/Maxim/OneDrive/Escritorio/Septimo semestre/Taller cortometraje/imgs/3.jpg");

	return 0;
}


