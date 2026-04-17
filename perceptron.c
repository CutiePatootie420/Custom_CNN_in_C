#include <stdio.h>
#include <stdlib.h>
typedef struct perceptron
{
    int bias;
    int input1;
    int input2;
    int output;
    struct perceptron* next;
}perceptron;
perceptron* init_basic_perceptron(int bias,int input1,int input2)
{
    perceptron* temp=malloc(sizeof(perceptron));
    temp->bias=bias;
    temp->input1=input1;
    temp->input2=input2;
    temp->output=input1+input2+bias;
    temp->next=NULL;
    return temp;
}
int main()
{
    perceptron* temp=init_basic_perceptron(3,-2,-2);
    printf("%d",temp->output);
    return 0;
}