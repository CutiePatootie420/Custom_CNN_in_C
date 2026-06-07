#include <stdlib.h>
#include "mlp.h"
#define IMAGE_SIZE 784
mlp* create_mlp(int* arr, int layers)
{
    mlp* temp=malloc(sizeof(mlp));
    temp->size=layers;
    temp->summary=malloc(sizeof(int)*layers); // Stores the number of nodes per layer
    temp->p_sums=malloc(sizeof(int)*layers); // Stores bias/activation offset prefix sums for flat 1D allocation
    temp->w_indices=malloc(sizeof(int)*(layers-1)); // Stores weight offsets mapping for transitions between layers
    
    for(int layer=0;layer<layers;layer++)
    {
        temp->summary[layer]=arr[layer];
        if(layer==0)
        {
            temp->p_sums[layer]=0;
            temp->w_indices[layer]=0;
        }
        else
        {
            if(layer!=layers-1)
            {
                // Weight offsets: cumulative sum of weights in previous layer transitions (nodes_current * nodes_previous)
                temp->w_indices[layer]=temp->w_indices[layer-1]+arr[layer]*arr[layer-1];
            }
            // Bias/activation offsets: prefix sum to map index to start of each layer's block
            temp->p_sums[layer]=temp->p_sums[layer-1]+arr[layer-1]; // indices of z,a,b,etc: arr[p_sums[layer]+node]
        }
    }
    // Calculate total allocation sizes for flat contiguous arrays
    temp->total_biases=temp->p_sums[layers-1]+arr[layers-1];
    temp->total_weights=temp->w_indices[layers-2]+arr[layers-1]*arr[layers-2];
    
    // Contiguous bias allocation (padded by IMAGE_SIZE to keep input indexing aligned)
    temp->biases=malloc(sizeof(double)*(temp->total_biases+IMAGE_SIZE)); 
    temp->weights=malloc(sizeof(double)*temp->total_weights);
    return temp;
}
static inline double random_weight()
{
    return (((double)rand()/RAND_MAX)-0.5); 
}
void initialise_network(mlp* temp)
{
    for(int i=0;i<temp->total_biases;i++)
    {
        temp->biases[i]=random_weight();
    }
    for(int i=0;i<temp->total_weights;i++)
    {
        temp->weights[i]=random_weight();
    }
}
void clear_network(mlp* temp)
{
    free(temp->biases);
    free(temp->weights);
    free(temp->p_sums);
    free(temp->summary);
    free(temp->w_indices);
    free(temp);
    return;
}