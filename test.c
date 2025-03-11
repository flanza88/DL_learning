// 1.3
double RoundUp(double N, int DecPlaces)
{
    int i;
    double AmountToAdd = 0.5;
    for (i = 0; i < DecPlaces; i++)
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
void PrintReal(double N, int DecPlaces)
{
    double FractionPart;
    int IntegerPart;
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