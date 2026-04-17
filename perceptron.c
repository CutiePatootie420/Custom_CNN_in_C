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
int main()
{
    int array[2]={1,1};
    printf("%d",run_basic_perceptron(nand_gate(),array,2));
    return 0;
}