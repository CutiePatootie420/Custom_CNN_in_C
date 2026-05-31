#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
typedef struct layer
{
    int size;
    double* activation;
    double* biases;
    double* z;
    double* weights;
    double* dc_da;
    double* dc_db;
    double* dc_dw;
}layer;
typedef struct mlp
{
    int size;
    struct layer* layers;
}mlp;
unsigned int read_big_endian_uint(FILE* fp)
{
    unsigned char bytes[4];
    fread(bytes, sizeof(unsigned char), 4, fp);
    return (bytes[0]<<24)|(bytes[1]<<16)|(bytes[2]<<8)|bytes[3];
}
mlp* create_mlp(int* arr, int num)
{
    mlp* temp=malloc(sizeof(mlp));
    temp->layers=malloc(sizeof(layer)*(num));
    temp->size=num;

    temp->layers[0].size=arr[0]; //l0
    temp->layers[0].activation=malloc(sizeof(double)*arr[0]);
    temp->layers[0].biases=NULL;
    temp->layers[0].z=NULL;
    temp->layers[0].dc_da=NULL;
    temp->layers[0].dc_db=NULL;
    temp->layers[0].weights=malloc(sizeof(double)*arr[0]*arr[1]);
    temp->layers[0].dc_dw=malloc(sizeof(double)*arr[0]*arr[1]);

    temp->layers[num-1].size=arr[num-1]; //l[output]
    temp->layers[num-1].activation=malloc(sizeof(double)*arr[num-1]);
    temp->layers[num-1].biases=malloc(sizeof(double)*arr[num-1]);
    temp->layers[num-1].weights=NULL;
    temp->layers[num-1].z=malloc(sizeof(double)*arr[num-1]);
    temp->layers[num-1].dc_da=malloc(sizeof(double)*arr[num-1]);
    temp->layers[num-1].dc_db=malloc(sizeof(double)*arr[num-1]);
    temp->layers[num-1].dc_dw=NULL;

    for(int layer=1;layer<temp->size-1;layer++) //hidden layers
    {
        temp->layers[layer].size=arr[layer];
        temp->layers[layer].activation=malloc(sizeof(double)*arr[layer]);
        temp->layers[layer].biases=malloc(sizeof(double)*arr[layer]);
        temp->layers[layer].z=malloc(sizeof(double)*arr[layer]);
        temp->layers[layer].weights=malloc(sizeof(double)*arr[layer]*arr[layer+1]);
        temp->layers[layer].dc_da=malloc(sizeof(double)*arr[layer]);
        temp->layers[layer].dc_db=malloc(sizeof(double)*arr[layer]);
        temp->layers[layer].dc_dw=malloc(sizeof(double)*arr[layer]*arr[layer+1]);
    }

    return temp;
}
void deallocate_network(mlp* temp)
{   
    for(int layer=0;layer<temp->size;layer++)
    {
        free(temp->layers[layer].activation);
        if(temp->layers[layer].biases!=NULL)
        {
            free(temp->layers[layer].biases);
        }
        if(temp->layers[layer].weights!=NULL)
        {
            free(temp->layers[layer].weights);
        }
        if(temp->layers[layer].z!=NULL)
        {
            free(temp->layers[layer].z);
        }
        if(temp->layers[layer].dc_da!=NULL)
        {
            free(temp->layers[layer].dc_da);
        }
        if(temp->layers[layer].dc_db!=NULL)
        {
            free(temp->layers[layer].dc_db);
        }
        if(temp->layers[layer].dc_dw!=NULL)
        {
            free(temp->layers[layer].dc_dw);
        }
    }
    free(temp->layers);
    free(temp);
}
double random_weight()
{
    return (((double)rand()/RAND_MAX)-0.5); 
}
void initialise_network(mlp* temp)
{
    for(int layer=0;layer<temp->size;layer++)
    {
        for(int node=0;node<temp->layers[layer].size;node++)
        {
            temp->layers[layer].activation[node]=random_weight();
            if(temp->layers[layer].biases!=NULL)
            {
                temp->layers[layer].biases[node]=random_weight();
            }
            if(temp->layers[layer].weights!=NULL)
            {
                for(int next_node=0;next_node<temp->layers[layer+1].size;next_node++)
                {
                    temp->layers[layer].weights[node*temp->layers[layer+1].size+next_node]=random_weight();
                }
            }
        }
    }
}
int main()
{
    srand(time(NULL));
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

    FILE* test_images_file=fopen("mnist/t10k-images-idx3-ubyte","rb"); //test images file
    if(test_images_file==NULL)
    {
        printf("Test file not found");
        return 1;
    }
    unsigned int test_magic_num,test_images_num,test_rows,test_cols;
    test_magic_num=read_big_endian_uint(test_images_file);
    test_images_num=read_big_endian_uint(test_images_file);
    test_rows=read_big_endian_uint(test_images_file);
    test_cols=read_big_endian_uint(test_images_file);
    unsigned char *test_pixels=malloc(sizeof(unsigned char)*test_rows*test_cols*test_images_num); //reading pixel values of test dataset into test_pixels var
    fread(test_pixels, sizeof(unsigned char), test_images_num * test_rows * test_cols, test_images_file);

    FILE* test_labels_file=fopen("mnist/t10k-labels-idx1-ubyte","rb"); //test labels file
    if(test_labels_file==NULL)
    {
        printf("Test labels file not found");
        return 1;
    }
    unsigned int test_labels_magic_num,test_labels_num;
    test_labels_magic_num=read_big_endian_uint(test_labels_file);
    test_labels_num=read_big_endian_uint(test_labels_file);
    unsigned char* test_labels=malloc(sizeof(unsigned char)*test_labels_num); //reading label values of test dataset into test_labels var
    fread(test_labels, sizeof(unsigned char), test_labels_num, test_labels_file);

    printf("Enter the number of hidden layers you want in your MLP for MNIST dataset: ");
    int num_layers; // input and output layers
    scanf("%d",&num_layers);
    num_layers+=2;
    int mlp_info[num_layers];
    mlp_info[0]=image_size;
    mlp_info[num_layers-1]=10;
    for(int i=1;i<num_layers-1;i++)
    {
        printf("Enter number of nodes in hidden layer %d: ",i-1);
        scanf("%d",&mlp_info[i]);
    }
    mlp* network=create_mlp(mlp_info,num_layers);
    initialise_network(network);


    deallocate_network(network);
    
    return 0;

}
