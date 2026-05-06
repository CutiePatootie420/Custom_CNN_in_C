#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef struct layer
{
    double* activation;
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

    unsigned int image_size=rows*cols;

    unsigned char *pixels=malloc(sizeof(unsigned char)*rows*cols*images_num); //reading pixel values of complete dataset into pixels var
    size_t read_count=fread(pixels,sizeof(unsigned char),images_num,train_images_file);
    if(read_count!=images_num)
    {
        printf("Read unsuccessfull\n");
        return 1;
    }

    int cnn_summary[]={image_size, 16,16,10}; //building cnn
    int num_layers=sizeof(cnn_summary)/sizeof(cnn_summary[0]);
    layer* cnn=malloc(sizeof(layer)*num_layers);
    for(int layer=0;layer<num_layers;layer++)
    {
        cnn[layer].activation=malloc(sizeof(double)*cnn_summary[layer]);
        cnn[layer].biases=malloc(sizeof(double)*cnn_summary[layer]);
        cnn[layer].weights=malloc(sizeof(double*)*cnn_summary[layer]);
        if(layer==num_layers-1)
        {
            for(int j=0;j<cnn_summary[layer];j++)
            {
                cnn[layer].weights[j]=NULL;
            }
        }
        else
        {
            for(int j=0;j<cnn_summary[layer];j++)
            {
                cnn[layer].weights[j]=malloc(sizeof(double)*cnn_summary[layer+1]);
            }
        }
    }

    for(int layer=0;layer<num_layers;layer++) //setting parameters to random
    {
        for(int node=0;node<cnn_summary[layer];node++)
        {
            cnn[layer].activation[node]=rand();
            cnn[layer].biases[node]=rand();
            
        }
        if(layer==num_layers-1)
        {
            for(int node=0;node<cnn_summary[layer];node++)
            {
                cnn[layer].weights[node]=NULL;
            }
        }
        else
        {
            for(int node=0;node<cnn_summary[layer];node++)
            {
                for(int next_node=0;next_node<cnn_summary[layer+1];next_node++)
                {
                    cnn[layer].weights[node][next_node]=rand();
                }
            }
        }
    }

    int iterations=10;
    int image_index;
    srand(time(NULL));
    for(int iter=0;iter<iterations;iter++) //selecting random images
    {
        image_index=abs(rand())%images_num;
        for(int pixel=0;pixel<image_size;pixel++) //loading image into cnn input layer
        {
            cnn[0].activation[pixel]=pixels[image_index*image_size+pixel];
        }
        

    }


    return 0;
}