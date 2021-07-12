#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

template<typename PixelType>
class Image
{
    using DataType = std::vector<std::vector<PixelType>>;

public:
    Image(){};
    void Read(const std::string& fileName);
    void Update();
private:
    DataType data;
    std::size_t rows{ 0 };
    std::size_t columns{ 0 };
};


template<typename  PixelType>
void Image<PixelType>::Read(const std::string& fileName)
{
    cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR);
    if (!image.data)
    {
        std::cerr<<"No image data\n";
        return;
    }
    rows = image.rows;
    columns = image.cols;

    data = DataType(rows, std::vector<PixelType>(columns, PixelType{}));

    PixelType pixel;
    for (unsigned r=0; r < rows; ++r)
    {
        cv::Vec3b * row = image.ptr<cv::Vec3b>(r);
        for (unsigned c=0; c < columns; ++c)
        {
            pixel[2] = row[c][2];  //R
            pixel[1] = row[c][1];  //G
            pixel[0] = row[c][0];  //B

            if constexpr (std::is_fundamental<PixelType>::value) //uchar, short, float (gray)
            {
                data[c][r] = static_cast<PixelType>((pixel[0] + pixel[1]+ pixel[2])/3);
            }
            else //RGB LAB, channels...
            {
                //memoria
                data[r][c][0] = static_cast<typename PixelType::ValueType>(pixel[0]);
                data[r][c][1] = static_cast<typename PixelType::ValueType>(pixel[1]);
                data[r][c][2] = static_cast<typename PixelType::ValueType>(pixel[2]);
            }
        }
    }
}

template<typename PixelType>
void Image<PixelType>::Update()
{
    cv::Mat image = cv::Mat::zeros(rows, columns, CV_8UC3);
    for(unsigned r=0 ; r<rows ; ++r)
    {
        for(unsigned c=0 ; c<columns ; ++c)
        {
            image.at<cv::Vec3b>(r,c)[0] = data[r][c][0];
            image.at<cv::Vec3b>(r,c)[1] = data[r][c][1];
            image.at<cv::Vec3b>(r,c)[2] = data[r][c][2];
        }
    }
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image", image);
    cv::waitKey();
}


#endif // IMAGE_H
