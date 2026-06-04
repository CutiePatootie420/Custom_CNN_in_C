#include "thread_handler.h"
#include <stdlib.h>
#include <math.h>
#define IMAGE_SIZE 784
state* initialise_state_arr(int threads, mlp* network,unsigned char* images, unsigned char* labels,int total_img,int img)
{
    state* temp=malloc(sizeof(state)*threads);
    for(int i=0;i<threads;i++)
    {
        temp[i].owner=network;
        temp[i].activation=malloc(sizeof(double)*(network->total_biases+IMAGE_SIZE));
        temp[i].dc_da=malloc(sizeof(double)*(network->total_biases+IMAGE_SIZE)); //first 784 elements of this arr are redundant but we add to maintain similarity of array access
        temp[i].dc_db=malloc(sizeof(double)*(network->total_biases+IMAGE_SIZE)); //first 784 elements of this arr are redundant but we add to maintain similarity of array access
        temp[i].z=malloc(sizeof(double)*(network->total_biases+IMAGE_SIZE)); //first 784 elements of this arr are redundant but we add to maintain similarity of array access
        temp[i].dc_dw=malloc(sizeof(double)*network->total_weights);
        temp[i].total_images=total_img;
        temp[i].img_per_state=img;
        temp[i].images=images;
        temp[i].labels=labels;
    }
    return temp;
}
void clear_state_arr(state* arr, int num)
{
    for(int i=0;i<num;i++)
    {
        free(arr[i].activation);
        free(arr[i].dc_da);
        free(arr[i].dc_db);
        free(arr[i].dc_dw);
        free(arr[i].z);
    }
    free(arr);
}
void load_image(double* activation,int idx, unsigned char* image)
{
    for(int node=0;node<IMAGE_SIZE;node++)
    {
        activation[node]=image[idx*IMAGE_SIZE+node]/255.0;
    }
}
double sigmoid_func(double x)
{
    return 1.0/(1.0+exp(-x));
}
void* feedforward_and_backprop(void* temp)
{
    state* arg=(state*)temp;
    mlp* net=arg->owner; //to eliminate pointer chasing killing performance
    int* p_sums=net->p_sums;
    int* w_indices=net->w_indices;
    for(int i=0;i<arg->img_per_state;i++)
    {
        int img_idx=abs(rand())%arg->total_images; //feedforward starts
        load_image(arg->activation,img_idx,arg->images);
        double softmax_temp=0;
        for(int layer=1;layer<net->size;layer++)
        {
            for(int node=0;node<net->summary[layer];node++)
            {
                double temp=0;
                for(int prev_node=0;prev_node<net->summary[layer-1];prev_node++)
                {
                    temp+=net->weights[w_indices[layer-1]+prev_node*net->summary[layer]+node]*arg->activation[p_sums[layer-1]+prev_node];
                }
                arg->z[p_sums[layer]+node]=temp+net->biases[net->p_sums[layer]+node];
                if(layer!=net->size-1)
                {
                    arg->activation[p_sums[layer]+node]=sigmoid_func(arg->z[p_sums[layer]+node]);
                }
                else 
                {
                    softmax_temp+=exp(arg->z[p_sums[layer]+node]);
                }
            }
        }
        int last_layer=net->size-1;
        for(int node=0;node<net->summary[last_layer];node++) //softmax activation for last layer nodes
        {
            arg->activation[p_sums[last_layer]+node]=exp(arg->z[p_sums[last_layer]+node])/softmax_temp;
        }
        
    }
    return NULL;
}





