#include <stdio.h>
#include <stdlib.h>
typedef struct perceptron
{
    double bias;
    double temp_sum;
    struct perceptron* next;
}perceptron;
perceptron* init_basic_perceptron(double bias)
{
    perceptron* temp=malloc(sizeof(perceptron));
    temp->bias=bias;
    temp->temp_sum=0;
    temp->next=NULL;
    return temp;
}
void run_basic_perceptron(double weight,double input, perceptron* temp)
{
    temp->temp_sum+=(weight*input);
}
void fire_perceptron(double weight,perceptron* curr)
{
    int output;
    if(curr->temp_sum+curr->bias>0)
    {
        output=1;
    }
    else
    {
        output=0;
    }
    run_basic_perceptron(weight,output,curr->next);
}
void bit_addition(int x1, int x2,int* sum, int* carry)
{
    double bias=3;
    double w=-2;
    perceptron* out1=init_basic_perceptron(bias);
    run_basic_perceptron(w,x1,out1);
    run_basic_perceptron(w,x2,out1);
    perceptron* out2=init_basic_perceptron(bias);
    out1->next=out2;
    run_basic_perceptron(w,x1,out2);
    fire_perceptron(w,out1);

    perceptron* out3=init_basic_perceptron(bias);
    run_basic_perceptron(w,x2,out3);
    out1->next=out3;
    fire_perceptron(w,out1);
    perceptron* out4=init_basic_perceptron(bias);
    out2->next=out4;
    fire_perceptron(w,out2);
    out3->next=out4;
    fire_perceptron(w,out3);
    perceptron* out5=init_basic_perceptron(bias);
    out1->next=out5;
    fire_perceptron(w,out1);
    fire_perceptron(w,out1);
    *sum=(out4->temp_sum+out4->bias)>0?1:0;
    *carry=(out5->temp_sum+out5->bias)>0?1:0;
}
int main()
{
    int x1=0;
    int x2=0;
    int sum;
    int carry;
    bit_addition(x1,x2,&sum,&carry);
    printf("%d %d",sum,carry);
    return 0;
}