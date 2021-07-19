#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <string>
#include "kernel.h"
#include <opencv2/opencv.hpp>
#include <cmath>

template<typename PixelType>
class Image
{
    using DataType = std::vector<std::vector<PixelType>>;

public:
    Image(){};
    void Read(const std::string& fileName);
    void Update();
    void toGrayscale();

    //Implementacion de filtros por Kernel
    void Filter(Kernel kernel, float scale);

    //Implementacion del los operadores de Sobel y Prewitt
    void Prewitt();
    void Sobel();
    //Implementacion de Mediana y Media
    std::vector<std::vector<PixelType>>  Mediana(DataType input, int rows, int cols);
    std::vector<std::vector<PixelType>>  Media(DataType input, int rows, int cols);

    //Ecualizacion
    std::vector<std::vector<PixelType>> Ecualizar(DataType input, int rows, int cols);


    //public Data
    DataType data;
    std::size_t rows{ 0 };
    std::size_t columns{ 0 };

private:
    int getMin(int array[], int size);
    cv::Mat fillBorders(cv::Mat& vec, int size);
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

template<typename PixelType>
PixelType getNuevoPixel(int row , int col , cv::Mat &filteredData, Kernel kernel, float scale)
{
    PixelType p;
    int padding = (kernel.size()-1)/2;
    int r=0,g=0,b=0;
    for(int i=0 ; i<kernel.size() ; i++)
    {
        for(int j=0 ; j<kernel.size() ; j++)
        {
            cv::Vec3b pix = filteredData.at<cv::Vec3b>(row+padding-1+i, col+padding-1+j);
            b += (kernel.data[i][j] * pix[0])*scale;
            g += (kernel.data[i][j] * pix[1])*scale;
            r += (kernel.data[i][j] * pix[2])*scale;
            p[0] = MAX(0, MIN(b , 255));     //BLUE
            p[1] = MAX(0, MIN(g , 255));     //GREEN
            p[2] = MAX(0, MIN(r , 255));     //RED

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
void Image<PixelType>::Filter(Kernel kernel, float scale)
{
    int borderSize = kernel.size()-1;
    cv::Mat paddingData = cv::Mat::zeros(rows+borderSize, columns+borderSize, CV_8UC3);
    paddingData = fillBorders(paddingData, borderSize/2);
    //Filtering
    for(int r=0 ; r<rows ; r++)
    {
        for(int c=0 ; c<columns ; c++)
        {
            PixelType pix = getNuevoPixel<PixelType>(r , c , paddingData, kernel, scale);
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
            data[r][c][0] = MIN(255, MAX(0 , gs));
            data[r][c][1] = MIN(255, MAX(0 , gs));
            data[r][c][2] = MIN(255, MAX(0 , gs));
        }
    }
}

template<typename PixelType>
void Image<PixelType>::Prewitt()
{
    Kernel prewittKernelX({
                              {-1,0,1},
                              {-1,0,1},
                              {-1,0,1}
                          });
    Kernel prewittKernelY({
                              {-1,-1,-1},
                              {0,0,0},
                              {1,1,1}
                          });
    float scale = 1.f/8.f;

    cv::Mat paddingDataX = cv::Mat::zeros(rows+2, columns+2, CV_8UC3);
    cv::Mat paddingDataY = cv::Mat::zeros(rows+2, columns+2, CV_8UC3);

    paddingDataX = fillBorders(paddingDataX, 1);
    paddingDataY = fillBorders(paddingDataY, 1);

    //Filtering
    for(int r=0 ; r<rows ; r++)
    {
        for(int c=0 ; c<columns ; c++)
        {
            PixelType pixX = getNuevoPixel<PixelType>(r , c , paddingDataX, prewittKernelX, scale);
            PixelType pixY = getNuevoPixel<PixelType>(r , c , paddingDataY, prewittKernelY, scale);
            data[r][c][0] = MIN(sqrt(pow(pixX[0],2) + pow(pixY[0],2)), 255);
            data[r][c][1] = MIN(sqrt(pow(pixX[0],2) + pow(pixY[1],2)), 255);
            data[r][c][2] = MIN(sqrt(pow(pixX[0],2) + pow(pixY[2],2)), 255);
        }
    }
}

template<typename PixelType>
void Image<PixelType>::Sobel()
{
    Kernel sobelKernelX({
                              {-1,0,1},
                              {-2,0,2},
                              {-1,0,1}
                          });
    Kernel sobelKernelY({
                              {-1,-2,-1},
                              {0,0,0},
                              {1,2,1}
                          });
    float scale = 1.f/6.f;

    cv::Mat paddingDataX = cv::Mat::zeros(rows+2, columns+2, CV_8UC3);
    cv::Mat paddingDataY = cv::Mat::zeros(rows+2, columns+2, CV_8UC3);

    paddingDataX = fillBorders(paddingDataX, 1);
    paddingDataY = fillBorders(paddingDataY, 1);

    //Filtering
    for(int r=0 ; r<rows ; r++)
    {
        for(int c=0 ; c<columns ; c++)
        {
            PixelType pixX = getNuevoPixel<PixelType>(r , c , paddingDataX, sobelKernelX, scale);
            PixelType pixY = getNuevoPixel<PixelType>(r , c , paddingDataY, sobelKernelY, scale);
            data[r][c][0] = MIN(sqrt(pow(pixX[0],2) + pow(pixY[0],2)), 255);
            data[r][c][1] = MIN(sqrt(pow(pixX[0],2) + pow(pixY[1],2)), 255);
            data[r][c][2] = MIN(sqrt(pow(pixX[0],2) + pow(pixY[2],2)), 255);
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

template<typename PixelType>
std::vector<std::vector<PixelType>>  Image<PixelType>::Mediana(DataType input, int rows, int cols)
{
    //Utilizaremos 3 bits para cada pixel
    DataType output = DataType(rows, std::vector<PixelType>(cols, PixelType{}));
    std::cout << rows << "," << cols << std::endl;
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            int n = 0;
            std::vector<int> Red;
            std::vector<int> Green;
            std::vector<int> Blue;

            for (int i = r - 1; i < r + 2; i++)
            {
                if (i < 0 || i > rows - 1)
                {
                    continue;
                }
                else
                {
                    for (int j = c - 1; j < c + 2; j++)
                    {
                        if (j < 0 || j > cols - 1)
                        {
                            continue;
                        }
                        else
                        {
                            n++;
                            Red.push_back(input[i][j][2]);
                            Green.push_back(input[i][j][1]);
                            Blue.push_back(input[i][j][0]);
                            /*sumR = sumR + input[i][j][2];
                            sumG = sumG + input[i][j][1];
                            sumB = sumB + input[i][j][0];*/
                        }
                    }
                }
            }
            std::sort(Red.begin(), Red.end());
            std::sort(Green.begin(), Green.end());
            std::sort(Blue.begin(), Blue.end());
            if(n%2==0)
            {
                output[r][c][2] = (Red[n / 2] + Red[(n / 2) + 1]) / 2;
                output[r][c][1] = (Green[n / 2] + Green[(n / 2) + 1]) / 2;
                output[r][c][0] = (Blue[n / 2] + Blue[(n / 2) + 1]) / 2;
            }
            else
            {
                output[r][c][2] = Red[(n-1) / 2];
                output[r][c][1] = Green[(n - 1) / 2];
                output[r][c][0] = Blue[(n - 1) / 2];
            }
        }
    }
    return output;
}

template<typename PixelType>
std::vector<std::vector<PixelType>>  Image<PixelType>::Media(DataType input, int rows, int cols)
{
    //Utilizaremos 3 bits para cada pixel
    DataType output = DataType(rows, std::vector<PixelType>(cols, PixelType{}));
    std::cout << rows << "," << cols << std::endl;
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            int n = 0;
            int sumR = 0;
            int sumG = 0;
            int sumB = 0;
            for (int i = r - 1; i < r + 2; i++)
            {
                if (i < 0 || i > rows - 1)
                {
                    continue;
                }
                else
                {
                    for (int j = c - 1; j < c + 2; j++)
                    {
                        if (j < 0 || j > cols - 1)
                        {
                            continue;
                        }
                        else
                        {
                            n++;
                            sumR = sumR + input[i][j][2];
                            sumG = sumG + input[i][j][1];
                            sumB = sumB + input[i][j][0];
                        }
                    }
                }
            }
            output[r][c][2] = sumR / n;
            output[r][c][1] = sumG / n;
            output[r][c][0] = sumB / n;
        }
    }
    return output;
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

#endif // IMAGE_H
