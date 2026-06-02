#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#if defined(_WIN32)
    #include <windows.h>
#elif defined(__APPLE__)
    #include <sys/param.h>
    #include <sys/sysctl.h>
#elif defined(_linux_) || defined(__unix__)
    #include <unistd.h>
#endif
typedef struct layer
{
    int size;
    double* biases;
    double* weights;
}layer;
typedef struct mlp
{
    int size;
    struct layer* layers;
}mlp;
typedef struct buffer
{
    double* activation;
    double* z;
    double* dc_da;
    double* dc_db;
    double* dc_dw;
}buffer;
typedef struct com_data
{
    mlp* nwk;
    buffer* state;
    unsigned char* images;
    unsigned char* labels;
    int images_per_t;
}com_data;
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
    temp->layers[0].biases=NULL;
    temp->layers[0].weights=malloc(sizeof(double)*arr[0]*arr[1]);

    temp->layers[num-1].size=arr[num-1]; //l[output]
    temp->layers[num-1].biases=malloc(sizeof(double)*arr[num-1]);
    temp->layers[num-1].weights=NULL;

    for(int layer=1;layer<temp->size-1;layer++) //hidden layers
    {
        temp->layers[layer].size=arr[layer];
        temp->layers[layer].biases=malloc(sizeof(double)*arr[layer]);
        temp->layers[layer].weights=malloc(sizeof(double)*arr[layer]*arr[layer+1]);
    }

    return temp;
}
void deallocate_network(mlp* temp)
{   
    for(int layer=0;layer<temp->size;layer++)
    {
        if(temp->layers[layer].biases!=NULL)
        {
            free(temp->layers[layer].biases);
        }
        if(temp->layers[layer].weights!=NULL)
        {
            free(temp->layers[layer].weights);
        }
    }
    free(temp->layers);
    free(temp);
}
static inline double random_weight()
{
    return (((double)rand()/RAND_MAX)-0.5); 
}
void initialise_network(mlp* temp)
{
    for(int layer=0;layer<temp->size;layer++)
    {
        for(int node=0;node<temp->layers[layer].size;node++)
        {
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
int get_cores()
{
    #if defined(_WIN32)
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        return (int)sysinfo.dwNumberOfProcessors;
    #elif defined(__APPLE__)
        int nm[2];
        int count=0;
        size_t len=sizeof(count);
        nm[0]=HW_NCPU;
        nm[1]=HW_AVAILCPU;
        sysctl(nm,2, &count, &len,NULL, 0);
        if(count<1)
        {
            nm[1]=HW_NCPU;
            sysctl(nm,2,&count,&len, NULL,0);
        }
        return count>0?count:1;
    #elif defined(_SC_NPROCESSORS_ONLN)
        return (int)sysconf(_SC_NPROCESSORS_ONLN);
    #else
        printf("OS not identified\n");
        return 1;
    #endif
}
buffer* initialise_buffer(int threads,int layers, int* temp)
{
    buffer* arr=malloc(sizeof(buffer)*threads);
    int a,z,da,db,dw;
    a=0;
    z=0;
    da=0;
    db=0;
    dw=0;
    for(int i=0;i<layers;i++)
    {
        a+=temp[i];
        z+=temp[i];
        da+=temp[i];
        db+=temp[i];
        if(i!=layers-1)
        {
            dw+=temp[i]*temp[i+1];
        }
    }
    for(int i=0;i<threads;i++)
    {
        arr[i].activation=malloc(sizeof(double)*a);
        arr[i].dc_da=malloc(sizeof(double)*da);
        arr[i].dc_db=malloc(sizeof(double)*db);
        arr[i].dc_dw=malloc(sizeof(double)*dw);
        arr[i].z=malloc(sizeof(double)*z);
    }
    return arr;
}
void* feedforward_backprop(void* arg)
{
    com_data* temp=(com_data*)arg;
    

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

    double learning_rate;
    int batch_size, epochs;
    printf("Enter desired learning rate: ");
    scanf("%lf",&learning_rate);
    printf("Enter desired bacth size (preferred like 2,4,16,32...): ");
    scanf("%d",&batch_size);
    printf("Enter desired epochs: ");
    scanf("%d",&epochs);
    int avl_cores=get_cores();
    int num_threads_ideal=(avl_cores>batch_size?batch_size:avl_cores);

    buffer* arr_buffer=initialise_buffer(num_threads_ideal,num_layers,mlp_info);
    int prefix_sums[num_layers];
    prefix_sums[0]=mlp_info[0];
    for(int i=1;i<num_layers;i++)
    {
        prefix_sums[i]=mlp_info[i]+prefix_sums[i-1];
    }
    com_data* t_package=malloc(sizeof(com_data)*num_threads_ideal);
    for(int i=0;i<num_threads_ideal;i++)
    {
        t_package[i].images=pixels;
        t_package[i].labels=labels;
        t_package[i].nwk=network;
        t_package[i].state=&arr_buffer[i];
        t_package[i].images_per_t=num_threads_ideal;
    }
    pthread_t threads[num_threads_ideal];

    int img_indices[images_num];
    for(int i=0;i<images_num;i++)
    {
        img_indices[i]=i;
    }
    for(int e=0;e<epochs;e++)
    {
        int batches=images_num/batch_size;
        for(int batch=0;batch<batches;batch++)
        {
            for(int t=0;t<num_threads_ideal;t++)
            {
                pthread_create(&threads[t],NULL,feedforward_backprop,&t_package[t]);
            }
        }
    }

    deallocate_network(network);

    return 0;

}
