#include <stdio.h>
#include <stdlib.h>
typedef struct perceptron
{
    int bias;
    int num_of_inputs;
    int* weights;
    struct perceptron* next;
}perceptron;
perceptron* init_basic_perceptron(int bias,int n,int* weight)
{
    perceptron* temp=malloc(sizeof(perceptron));
    temp->bias=bias;
    temp->num_of_inputs=n;
    temp->weights=malloc(sizeof(int)*temp->num_of_inputs);
    for(int i=0;i<n;i++)
    {
        temp->weights[i]=weight[i];
    }
    temp->next=NULL;
    return temp;
}
perceptron* nand_gate()
{
    int weights[2]={-2,-2};
    return init_basic_perceptron(3,2,weights);
}
int run_basic_perceptron(perceptron* temp,int* input, int n)
{
    int x=0;
    for(int i=0;i<n;i++)
    {
        x+=(temp->weights[i]*input[i]);
    }
    return (x+(temp->bias))<=0?0:1;
}
int run_nand_gate(int* input)
{
    return run_basic_perceptron(nand_gate(),input,2);
}
void bit_addition(int* x, int* sum, int* carry)
{
    int out1=run_nand_gate(x);
    int in2[2]={x[0],out1};
    int out2=run_nand_gate(in2);
    int in3[2]={out1,x[1]};
    int out3=run_nand_gate(in3);
    int in4[2]={out2,out3};
    int out4=run_nand_gate(in4);
    int in5[2]={out1,out1};
    int out5=run_nand_gate(in5);
    *sum=out4;
    *carry=out5;
    
}
int main()
{
    int array[2]={0,1};
    int sum;
    int carry;
    bit_addition(array,&sum,&carry);
    printf("%d %d",sum,carry);
    return 0;
}