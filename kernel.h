#ifndef KERNEL_H
#define KERNEL_H

#include <iostream>
#include <vector>

class Kernel
{
    using DataType = std::vector<std::vector<float>>;
    int tam;
public:
    DataType data;
    Kernel(int s)
    {
        tam = s;
        data = DataType(s, std::vector<float>(s, {}));
    }
    Kernel(DataType d)
    {
        if(d.size() == d[0].size() && d.size()%2!=0)
        {
            tam = d.size();
            data = d;
        }
        else
        {
            std::cout<<"Kernel no cuadrado"<<std::endl;
        }
    }
    int size()
    {
        return tam;
    }
    void showKernel()
    {
        for(int i=0 ; i<tam ; i++)
        {
            for(int j=0 ; j<tam ; j++)
                std::cout<<data[i][j]<<"\t";
            std::cout<<std::endl;
        }
    }
};

#endif // KERNEL_H
