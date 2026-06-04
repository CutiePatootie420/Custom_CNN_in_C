#include "mlp.h"
#pragma once
typedef struct state
{
    mlp* owner;
    unsigned char* images;
    unsigned char* labels;
    int total_images;
    int img_per_state;
    double* activation;
    double* z;
    double* dc_da;
    double* dc_db;
    double* dc_dw;
}state;
state* initialise_state_arr(int threads, mlp* network,unsigned char* images, unsigned char* labels,int total_img,int img);
void clear_state_arr(state* arr, int num);
void* feedforward_and_backprop(void* temp);
