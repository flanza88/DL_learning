# 1.4Python基础


# code1-1通过牛顿迭代法求解平方根
def squareroot(n):
    root = n / 2
    for k in range(20):
        root = (1 / 2) * (root + n / root)
    return root


print(squareroot(9))
print(squareroot(4563))


# code1-2Fraction类及其构造方法
class Fraction:
    def __init__(self, top, bottom):
        self.num = top
        self.den = bottom  # 分母denominator

    # code1-3show方法
    def show(self):
        print(self.num, "/", self.den)

    # code1-4__str__方法
    # 实例化Fraction后，可以打印以下内容
    def __str__(self):
        return str(self.num) + "/" + str(self.den)

    # code1-5__add__方法
    # 实例化Fraction后，可以通过对象名相加
    def __add__(self, otherfraction):
        newnum = self.num * otherfraction.den + self.den * otherfraction.num
        newden = self.den * otherfraction.den
        # code1-7改良版__add__方法
        common = gcd(newnum, newden)
        return Fraction(newnum // common, newden // common)

    # code1-8__eq__方法
    def __eq__(self, other):
        firstnum = self.num * other.den
        secondnum = other.num * self.den
        return firstnum == secondnum


# code1-6gcd函数
def gcd(m, n):
    while m % n != 0:
        oldm = m
        oldn = n
        m = oldn
        n = oldm % oldn
    return n


myf = Fraction(3, 5)
print(myf)
myf.show()
print("I ate", myf, "of the pizza")
f1 = Fraction(1, 4)
f2 = Fraction(1, 2)
print(f1 + f2)


# code1-10超类LogicGate
class LogicGate:
    def __init__(self, n):
        self.label = n
        self.output = None

    def getLabel(self):
        return self.label

    def getOutput(self):
        self.output = self.performGateLogic()
        return self.output


# code1-11BinaryGate类
class BinaryGate(LogicGate):
    def __init__(self, n):
        super().__init__(n)
        self.pinA = None
        self.pinB = None

    def getPinA(self):
        return int(input("Enter Pin A input for gate " + self.getLabel() + "-->"))

    def getPinB(self):
        return int(input("Enter Pin B input for gate " + self.getLabel() + "-->"))


# code1-12UnaryGate类
class UnaryGate(LogicGate):
    def __init__(self, n):
        super().__init__(n)
        self.pin = None

    def getPin(self):
        return int(input("Enter Pin input for gate " + self.getLabel() + "-->"))


# code1-13AndGate类
class AndGate(BinaryGate):
    def __init__(self, n):
        super().__init__(n)

    def performGateLogic(self):
        a = self.getPinA()
        b = self.getPinB()
        if a == 1 and b == 1:
            return 1
        else:
            return 0
