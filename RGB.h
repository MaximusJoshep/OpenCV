#pragma once
#include <vector>
template<typename Type>
class RGB
{
public:
	RGB();
	const Type& operator[](unsigned index)const { return Channels[index];}
	Type& operator[](unsigned index){ return Channels[index];}

	using ValueType = Type;
private:
	std::vector<Type> Channels{};
};
template<typename Type>
RGB<Type>::RGB() :Channels{ std::vector<Type>(3,{})}
{

}
