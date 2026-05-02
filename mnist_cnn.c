#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "node.h"
typedef struct layer
{
    unsigned char* gs_vals;
    double* biases;
    double** weights;

}layer;
unsigned int read_big_endian_uint(FILE* fp)
{
    unsigned char bytes[4];
    fread(bytes, sizeof(unsigned char), 4, fp);
    return (bytes[0]<<24)|(bytes[1]<<16)|(bytes[2]<<8)|bytes[3];
}
int main()
{
    FILE* train_images_file=fopen("mnist/train-images-idx3-ubyte","rb");
    if(train_images_file==NULL)
    {
        printf("Read failed\n");
        return 1;
    }
    unsigned int magic_num,images_num,rows,cols;
    magic_num=read_big_endian_uint(train_images_file);
    images_num=read_big_endian_uint(train_images_file);
    rows=read_big_endian_uint(train_images_file);
    cols=read_big_endian_uint(train_images_file);
    printf("%u %u %u\n",images_num,rows,cols);
    unsigned int image_size=rows*cols;
    unsigned char *pixels=malloc(sizeof(unsigned char)*rows*cols*images_num);
    srand(time(NULL));
    int cnn_info[]={rows*cols,16,16,10};
    layer* layers;
    layers=malloc(sizeof(layer)*(sizeof(cnn_info)/sizeof(cnn_info[0])));
    for(int i=0;i<sizeof(cnn_info)/sizeof(cnn_info[0]);i++)
    {
        layers[i].gs_vals=malloc(sizeof(unsigned char)*cnn_info[i]);
        layers[i].biases=malloc(sizeof(double)*cnn_info[i]);
        if(i==sizeof(cnn_info)/sizeof(cnn_info[0]) -1)
        {
            layers[i].weights=NULL;
        }
        else
        {
            layers[i].weights=malloc(sizeof(double*)*cnn_info[i]);
            for(int j=0;j<cnn_info[i];j++)
            {
                layers[i].weights[j]=malloc(sizeof(double)*cnn_info[i+1]);
            }
        }
    }
    for(int i=0;i<)
    for(int i=0;i<sizeof(cnn_info)/sizeof(cnn_info[0]);i++)
    {
        free(layers[i].gs_vals);
        free(layers[i].biases);
        if(i==sizeof(cnn_info)/sizeof(cnn_info[0]) -1)
        {
        }
        else
        {
            for(int j=0;j<cnn_info[i];j++)
            {
                free(layers[i].weights[j]);
            }
            free(layers[i].weights);

        }
    }
    free(pixels);
    return 0;
}