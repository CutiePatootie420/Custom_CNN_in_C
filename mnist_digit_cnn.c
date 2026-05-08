#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
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
void feed_forward(double* activation_layer, int first_layer_num, double** weights_mtx, int second_layer_num, double* target_layer,double* bias_layer)
{
    double temp;
    for(int i=0;i<second_layer_num;i++)
    {
        temp=0;
        for(int j=0;j<first_layer_num;j++)
        {
            temp+=activation_layer[j]*weights_mtx[j][i];
        }
        target_layer[i]=temp+bias_layer[i];
    }
}
void sigmoid_func(double* target_layer, int layer_size)
{
    for(int i=0;i<layer_size;i++)
    {
        target_layer[i]=1.0/(1.0+exp(-target_layer[i]));
    }
}
double random_weight()
{
    return (((double)rand()/RAND_MAX)-0.5)*0.1;
}
int main()
{
    FILE* train_images_file=fopen("mnist/train-images-idx3-ubyte","rb"); //image file
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

    FILE* label_images_file=fopen("mnist/train-labels-idx1-ubyte","rb"); //label file
    if(label_images_file==NULL)
    {
        printf("Read failed\n");
        return 1;
    }
    unsigned int label_magic_num,label_num;
    label_magic_num=read_big_endian_uint(label_images_file);
    label_num=read_big_endian_uint(label_images_file);


    unsigned int image_size=(size_t)rows*cols;

    unsigned char *pixels=malloc(sizeof(unsigned char)*rows*cols*images_num); //reading pixel values of complete dataset into pixels var
    size_t image_read_count=fread(pixels,sizeof(unsigned char),images_num*rows*cols,train_images_file);
    if(image_read_count!=images_num*rows*cols)
    {
        printf("Read unsuccessfull\n");
        return 1;
    }

    unsigned char* labels=malloc(sizeof(unsigned char)*label_num); //reading label values of complete dataset into labels var
    size_t label_read_count=fread(labels,sizeof(unsigned char),label_num,label_images_file);
    if(label_read_count!=label_num)
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
        
        if(layer==num_layers-1)
        {
            cnn[layer].weights=NULL;
        }
        else
        {
            cnn[layer].weights=malloc(sizeof(double*)*cnn_summary[layer]);
            for(int j=0;j<cnn_summary[layer];j++)
            {
                cnn[layer].weights[j]=malloc(sizeof(double)*cnn_summary[layer+1]);
            }
        }
    }

    srand(time(NULL));

    for(int layer=0;layer<num_layers;layer++) //setting parameters to random
    {
        for(int node=0;node<cnn_summary[layer];node++)
        {
            cnn[layer].activation[node]=random_weight();
            cnn[layer].biases[node]=random_weight();
        }
        if(layer!=num_layers-1)
        {
            for(int node=0;node<cnn_summary[layer];node++)
            {
                for(int next_node=0;next_node<cnn_summary[layer+1];next_node++)
                {
                    cnn[layer].weights[node][next_node]=random_weight();
                }
            }
        }
    }

    int iterations=5;
    int image_index;

    double* cost=malloc(sizeof(double)*cnn_summary[num_layers-1]);
    double final_cost;

    for(int iter=0;iter<iterations;iter++) //selecting random images
    {
        image_index=abs(rand())%images_num;
        for(int pixel=0;pixel<image_size;pixel++) //loading image into cnn input layer
        {
            cnn[0].activation[pixel]=pixels[image_index*image_size+pixel]/255.0;
        }
        
        for(int i=0;i<num_layers-1;i++) //processing cnn layers
        {
            int j=i+1;
            feed_forward(cnn[i].activation,cnn_summary[i],cnn[i].weights,cnn_summary[j],cnn[j].activation,cnn[j].biases);
            sigmoid_func(cnn[j].activation,cnn_summary[j]);
        }

        for(int i=0;i<cnn_summary[num_layers-1];i++)
        {
            printf("%d: %f\n",i,cnn[num_layers-1].activation[i]);
        }

        final_cost=0; 
        for(int i=0;i<cnn_summary[num_layers-1];i++) //cost 
        {
            if(labels[image_index]==i)
            {
                cost[i]=(1-cnn[num_layers-1].activation[i])*(1-cnn[num_layers-1].activation[i]);
            }
            else
            {
                cost[i]=(cnn[num_layers-1].activation[i])*cnn[num_layers-1].activation[i];
            }
            final_cost+=cost[i];
        }
        
    }


    return 0;
}