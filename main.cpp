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

        Kernel k({
                    {-1,0,1},
                    {-1,0,1},
                    {-1,0,1},
                });

        image.toGrayscale();
        image.update();
        image.filter(k);
        image.update();
    }
    else
    {
        std::cout<<"Ingrese la direccion de la imagen(solo un parametro)"<<std::endl;
    }
    return 0;
}
