#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <string>
#include "kernel.h"
#include <opencv2/opencv.hpp>

template<typename PixelType>
class Image
{
    using DataType = std::vector<std::vector<PixelType>>;

public:
    Image(){};
    void Read(const std::string& fileName);
    void update();
    void filter(Kernel kernel);
    void toGrayscale();
    void Mediana(cv::Mat input, cv::Mat output);
    std::vector<std::vector<PixelType>> Ecualizar(DataType input, int rows, int cols);
private:
    int getMin(int array[], int size);
    cv::Mat fillBorders(cv::Mat& vec, int size);
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
void Image<PixelType>::update()
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

template<typename PixelType>
PixelType getNuevoPixel(int row , int col , cv::Mat &filteredData, Kernel kernel)
{
    PixelType p;
    int padding = (kernel.size()-1)/2;
    int kernelTotalSize = kernel.size()*kernel.size();
    for(int i=0 ; i<kernel.size() ; i++)
    {
        for(int j=0 ; j<kernel.size() ; j++)
        {
            cv::Vec3b pix = filteredData.at<cv::Vec3b>(row+padding, col+padding);
            p[0] = p[0] + (kernel.data[i][j] * pix[0])/kernelTotalSize;
            p[1] = p[1] + (kernel.data[i][j] * pix[1])/kernelTotalSize;
            p[2] = p[2] + (kernel.data[i][j] * pix[2])/kernelTotalSize;
            p[0] = MIN(p[0] , 254);     //BLUE
            p[1] = MIN(p[1] , 254);     //GREEN
            p[2] = MIN(p[2] , 254);     //RED
        }
    }
    return p;
}

template<typename PixelType>
cv::Mat Image<PixelType>::fillBorders(cv::Mat& vec, int size)
{
    //CornerBorders border: TOP-LEFT
    for(int i=0 ; i<size ; i++)
    {
        for(int j=0 ; j<size ; j++)
        {
            vec.at<cv::Vec3b>(i,j)[0] = data[0][0][0];
            vec.at<cv::Vec3b>(i,j)[1] = data[0][0][1];
            vec.at<cv::Vec3b>(i,j)[2] = data[0][0][2];
        }
    }
    //CornerBorders border: TOP-RIGHT
    for(int i=0 ; i<size ; i++)
    {
        for(int j=0 ; j<size ; j++)
        {
            vec.at<cv::Vec3b>(rows+size+i,j)[0] = data[0][columns-1][0];
            vec.at<cv::Vec3b>(rows+size+i,j)[1] = data[0][columns-1][1];
            vec.at<cv::Vec3b>(rows+size+i,j)[2] = data[0][columns-1][2];
        }
    }
    //CornerBorders border: BOTTOM-RIGHT
    for(int i=0 ; i<size ; i++)
    {
        for(int j=0 ; j<size ; j++)
        {
            vec.at<cv::Vec3b>(rows+size+i,columns+size+j)[0] = data[rows-1][columns-1][0];
            vec.at<cv::Vec3b>(rows+size+i,columns+size+j)[1] = data[rows-1][columns-1][1];
            vec.at<cv::Vec3b>(rows+size+i,columns+size+j)[2] = data[rows-1][columns-1][2];
        }
    }
    //CornerBorders border: BOTTOM-LEFT
    for(int i=0 ; i<size ; i++)
    {
        for(int j=0 ; j<size ; j++)
        {
            vec.at<cv::Vec3b>(i,columns+size+j)[0] = data[rows-1][0][0];
            vec.at<cv::Vec3b>(i,columns+size+j)[1] = data[rows-1][0][1];
            vec.at<cv::Vec3b>(i,columns+size+j)[2] = data[rows-1][0][2];
        }
    }

    //Top borders
    for(int i=0 ; i<size ; i++)
    {
        for(int j=0 ; j<columns ; j++)
        {
            vec.at<cv::Vec3b>(i,j+size)[0] = data[0][j][0];
            vec.at<cv::Vec3b>(i,j+size)[1] = data[0][j][1];
            vec.at<cv::Vec3b>(i,j+size)[2] = data[0][j][2];
        }
    }
    //Bottom borders
    for(int i=0 ; i<size ; i++)
    {
        for(int j=0 ; j<columns ; j++)
        {
            vec.at<cv::Vec3b>(i+rows+size,j+size)[0] = data[rows-1][j][0];
            vec.at<cv::Vec3b>(i+rows+size,j+size)[1] = data[rows-1][j][1];
            vec.at<cv::Vec3b>(i+rows+size,j+size)[2] = data[rows-1][j][2];
        }
    }
    //Left borders
    for(int i=0 ; i<rows ; i++)
    {
        for(int j=0 ; j<size ; j++)
        {
            vec.at<cv::Vec3b>(i+size,j)[0] = data[i][0][0];
            vec.at<cv::Vec3b>(i+size,j)[1] = data[i][0][1];
            vec.at<cv::Vec3b>(i+size,j)[2] = data[i][0][2];
        }
    }
    //Right borders
    for(int i=0 ; i<rows ; i++)
    {
        for(int j=0 ; j<size ; j++)
        {
            vec.at<cv::Vec3b>(i+size,j+columns+size)[0] = data[i][columns-1][0];
            vec.at<cv::Vec3b>(i+size,j+columns+size)[1] = data[i][columns-1][1];
            vec.at<cv::Vec3b>(i+size,j+columns+size)[2] = data[i][columns-1][2];
        }
    }

    //Filling image
    for(int i=0 ; i<rows ; i++)
    {
        for(int j=0 ; j<columns ; j++)
        {
            vec.at<cv::Vec3b>(i+size,j+size)[0] = data[i][j][0];
            vec.at<cv::Vec3b>(i+size,j+size)[1] = data[i][j][1];
            vec.at<cv::Vec3b>(i+size,j+size)[2] = data[i][j][2];
        }
    }

    return vec;
}

template<typename PixelType>
void Image<PixelType>::filter(Kernel kernel)
{
    int borderSize = kernel.size()-1;
    cv::Mat filteredData = cv::Mat::zeros(rows+borderSize, columns+borderSize, CV_8UC3);
    filteredData = fillBorders(filteredData, borderSize/2);
    //Filtering
    for(int r=0 ; r<rows ; r++)
    {
        for(int c=0 ; c<columns ; c++)
        {
            PixelType pix = getNuevoPixel<PixelType>(r , c , filteredData, kernel);
            data[r][c] = pix;
        }
    }
}
template<typename PixelType>
void Image<PixelType>::toGrayscale()
{
    for(int r=0 ; r<rows ; r++)
    {
        for(int c=0 ; c<columns ; c++)
        {
            PixelType pixel = data[r][c];
            typename PixelType::ValueType gs = (pixel[0] + pixel[1] + pixel[2])/3;
            data[r][c][0] = gs;
            data[r][c][1] = gs;
            data[r][c][2] = gs;
        }
    }
}

template<typename PixelType>
std::vector<std::vector<PixelType>> Image<PixelType>::Ecualizar(DataType input, int rows, int cols)
{
    DataType output = DataType(rows, std::vector<PixelType>(cols, PixelType{}));
    // Calculamos el histograma.
    int hist[256];
    for (int i = 0; i < 256; i++) {
        hist[i] = 0;
    }
    for (int r = 0; r < rows; r++)
    {
       for (int c = 0; c < cols; c++)
       {
           int ind = input[r][c][0];
           hist[ind] = hist[ind] + 1;
       }
    }
    // Calculamos el histograma acumulado
    int accum_hist[256];
    accum_hist[0] = hist[0];
    for (int i = 1; i < 256; i++) {
        accum_hist[i] = hist[i] + accum_hist[i - 1];
    }
    int accumMinimum = getMin(accum_hist, 256);
    int lookUp[256];
    for (int i = 0; i < 256; i++) {
        lookUp[i] = floor(255 * (accum_hist[i] - accumMinimum) / (rows * cols - accumMinimum));
    }
    // Ecualizamos la imagen
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {

            output[r][c][0] = lookUp[input[r][c][0]];
            output[r][c][1] = lookUp[input[r][c][1]];
            output[r][c][2] = lookUp[input[r][c][2]];
        }
    }
    /*
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image", output);
    cv::waitKey();
    */

   return output;
}

template<typename PixelType>
void Image<PixelType>::Mediana(cv::Mat input, cv::Mat output)
{
    output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
    int windowSize = 3;
    int windowRad = windowSize/2;
    /* Expand image filling with zeros */
    cv::Mat aux = cv::Mat::zeros(input.rows + 2 * windowRad, input.cols + 2 * windowRad, CV_32FC1);
    cv::Mat ROI = aux(cv::Rect(windowRad, windowRad, input.cols, input.rows));
    // Rect (x,y,width ,height )
    input.copyTo(ROI);
    for (int r = windowRad; r < aux.rows - windowRad; r++)
    {
        for (int c = windowRad; c < aux.cols - windowRad; c++)
        {
            std::vector <float> pixVals;
            for (int m_r = 0; m_r < windowSize; m_r++)
            {
                for (int m_c = 0; m_c < windowSize; m_c++)
                {
                    pixVals.push_back(aux.at <float >(r + m_r - windowRad, c + m_c - windowRad));
                }
            }
            std::sort (&pixVals[0], &pixVals[(pixVals.size() - 1)]);
            output.at <float >(r - windowRad, c - windowRad) = pixVals[(pixVals.size() - 1) / 2];
            // Set the median to corresponding output pixel .
        }
    }
}

template<typename PixelType>
int Image<PixelType>::getMin(int array[], int size)
{
    int min = 0;
    for (int i = 0; i < size; i++)
        if (array[i] < min)
            min = array[i];
    return min;
}

#endif // IMAGE_H
