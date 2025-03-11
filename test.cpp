#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<iostream>

//串的顺序存储
#define MAXLEN 255
typedef struct{
    char ch[MAXLEN]
    int length;
}SString;//Static String
typedef struct{
    char * ch;
    int length;
}HString;//Heap String
HString S;
S.ch = (char*)malloc(MAXLEN*sizeof(char));
S.length = 0;

bool SubString(SString &Sub, SString S, int pos, int len){
    if(pos + len - 1 > S.length)
        return false;
    for(int i = pos; i < pos +len; i++)
        Sub.ch[i - pos + 1] = S.ch[i];//Sub节点存储的串值
    Sub.length = len;//Sub节点存储的串长
}
int StrCompare(SString S, SString T){
    for(int i = 1; i <= S.length && i <= T.length; i++){//第0位不要了，相当于位序=下标
        if(S.ch[i] != T.ch[i])
            return S.ch[i] - T.ch[i];
    }
    return S.length - T.length;
}
int Index(SString S, SString T){//T是子串
    int i = 1, j = 1;
    SString sub;
    while(i <= S.length && j <= T.length){
        if(S.ch[i] == T.ch[j]){
            i++;
            j++;
        }
        else{
            i = i - j + 2;//i回退到T串的起始位置
            j = 1;
        }
    }
    if(j > T.length)
        return i - T.length;//i回退到T串的起始位置
    else
        return 0;
}
//串的链式存储
typedef struct StringNode{
    char ch[4];//一个字符但那只存1个字符的话，存储密度低
    struct StringNode* next;
}StringNode, * String;

