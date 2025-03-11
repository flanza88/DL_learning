#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

// 1.3
int IntPart(double N)//��ȡ��������
{
    return (int)N;
}
double DecPart(double N)//��ȡС������
{
	return N - IntPart(N);
}
void PrintDigit(int ADigit)
{
	putchar(ADigit + '0');//ʵ������������ַ�
}
void PrintOut(int IntegerPart)
{
	printf("%d", IntegerPart);
}
double RoundUp(double N, int DecPlaces)//��������
{
    int i;
    double AmountToAdd = 0.5;
    for (i = 0; i < DecPlaces; i++)//����1λС��ֱ��
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
void PrintReal(double N, int DecPlaces)//��ӡʵ��
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
	int DecPlaces = 3;//С�������3λ
	PrintReal(number, DecPlaces);
    return 0;
}