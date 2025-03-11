#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

// 1.3
int IntPart(double N)//获取整数部分
{
    return (int)N;
}
double DecPart(double N)//获取小数部分
{
	return N - IntPart(N);
}
void PrintDigit(int ADigit)
{
	putchar(ADigit + '0');//实际上是输出了字符
}
void PrintOut(int IntegerPart)
{
	printf("%d", IntegerPart);
}
double RoundUp(double N, int DecPlaces)//四舍五入
{
    int i;
    double AmountToAdd = 0.5;
    for (i = 0; i < DecPlaces; i++)//保留1位小数直接
    {
        AmountToAdd /= 10;
    }
    return N + AmountToAdd;
}
void PrintFractionPart(double FractionPart, int DecPlaces)
{
    int i, Adigit;

    for (i = 0; i < DecPlaces; i++)
    {
        FractionPart *= 10;
        Adigit = IntPart(FractionPart);
        PrintDigit(Adigit);
        FractionPart = DecPart(FractionPart);
    }
}
void PrintReal(double N, int DecPlaces)//打印实数
{
    int IntegerPart;
    double FractionPart;
    if (N < 0)
    {
        putchar('-');
        N = -N;
    }
    N = RoundUp(N, DecPlaces);
    IntegerPart = IntPart(N);
    FractionPart = DecPart(N);
    PrintOut(IntegerPart);
    if (DecPlaces > 0)
        putchar('.');
    PrintFractionPart(FractionPart, DecPlaces);
}
int main()
{
    double number = -123.456789;
	int DecPlaces = 3;//小数点后保留3位
	PrintReal(number, DecPlaces);
    return 0;
}