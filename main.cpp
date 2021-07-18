#include <iostream>
#include "image.h"
#include "rgb.h"
#include "kernel.h"

int main(int argc, char* argv[])
{
    if(argc > 1)
    {
        std::cout<<argv[1]<<std::endl;
        std::string image_path = argv[1];
        Image<RGB<u_char>>image;
        image.Read(image_path);

        /*
        Kernel smoothKernel({
            {1,1,1},
            {1,1,1},
            {1,1,1}
        });
        */


        //Grayscale
        image.toGrayscale();
        image.Update();

        /*
        //Smooth
        float scale = 1/9.f;
        image.Filter(smoothKernel, scale);
        image.Update();
        */

        //Prewitt
        //image.Prewitt();
        //image.Update();

        //Sobel
        image.Sobel();
        image.Update();

    }
    else
    {
        std::cout<<"Ingrese la direccion de la imagen(solo un parametro)"<<std::endl;
    }
    return 0;
}
