#ifndef RGB_H
#define RGB_H

#include <vector>

template<typename Type>
class RGB
{
    std::vector<Type> Channels{};
public:
    RGB();
    const Type& operator[](unsigned index) const
    {
        return Channels[index];
    }
    Type& operator[](unsigned index)
    {
        return Channels[index];
    }
    using ValueType = Type;
};

template<typename Type>
RGB<Type>::RGB(): Channels{std::vector<Type>(3, {})}
{

}
#endif // RGB_H
