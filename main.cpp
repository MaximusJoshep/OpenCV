#include <iostream>
#include "image.h"
#include "rgb.h"
int main(int argc, char* argv[])
{
    if(argc > 1)
    {
        std::cout<<argv[1]<<std::endl;
        std::string image_path = argv[1];
        Image<RGB<u_char>>image;
        image.Read(image_path);
        image.Update();

    }
    else
    {
        std::cout<<"Ingrese la direccion de la imagen(solo un parametro)"<<std::endl;
    }
    return 0;
}
