#include <stdio.h>
#include "file_handler.h"
#include "mlp.h"
#include "thread_handler.h"
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#define IMAGE_SIZE 784
#define MAX_PIXEL_VAL 255
#if defined(_WIN32)
    #include <windows.h>
#elif defined(__APPLE__)
    #include <sys/param.h>
    #include <sys/sysctl.h>
#elif defined(_linux_) || defined(__unix__)
    #include <unistd.h>
#endif
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
int main()
{
    srand(time(NULL));
    image_file train_images_file=read_image_file("mnist/train-images-idx3-ubyte"); 
    if(train_images_file.file==NULL)
    {
        return 1;
    }
    label_file train_labels_file=read_label_file("mnist/train-labels-idx1-ubyte");

    unsigned char* train_images_pixels=malloc(sizeof(unsigned char)*train_images_file.images_num*IMAGE_SIZE); //training image pixel values
    if(fread(train_images_pixels,sizeof(unsigned char),IMAGE_SIZE*train_images_file.images_num,train_images_file.file)!=IMAGE_SIZE*train_images_file.images_num)
    {
        printf("Read unsuccessfull\n");
        return 1;
    }
    unsigned char* train_labels=malloc(sizeof(unsigned char)*train_labels_file.labels_num); //training set labels
    if(fread(train_labels,sizeof(unsigned char),train_labels_file.labels_num,train_labels_file.file)!=train_labels_file.labels_num)
    {
        printf("Read unsuccessfull\n");
        return 1;
    }

    printf("Enter the number of hidden layers you want in your MLP for MNIST dataset: ");
    int num_layers; // input and output layers
    scanf("%d",&num_layers);
    num_layers+=2;
    int mlp_info[num_layers];
    mlp_info[0]=IMAGE_SIZE;
    mlp_info[num_layers-1]=10;
    for(int i=1;i<num_layers-1;i++)
    {
        printf("Enter number of nodes in hidden layer %d: ",i-1);
        scanf("%d",&mlp_info[i]);
    }
    mlp* network=create_mlp(mlp_info, num_layers);
    initialise_network(network); //set weights and activations to random, maintaining xavier-glorot range

    double learning_rate;
    int batch_size, epochs;
    printf("Enter desired learning rate: ");
    scanf("%lf",&learning_rate);
    printf("Enter desired bacth size (preferred like 2,4,16,32...): ");
    scanf("%d",&batch_size);
    printf("Enter desired epochs: ");
    scanf("%d",&epochs);
    int avl_cores=get_cores();
    int num_threads_ideal=(avl_cores>batch_size?batch_size:avl_cores); //number of threads is preferred to be equal to avl cores
    printf("avl cores: %d\n",avl_cores);

    state* buffer_arr=initialise_state_arr(num_threads_ideal, network,train_images_pixels,train_labels,train_images_file.images_num,0); //number of state buffers is equal to threads, each buffer will be passed to a thread
    pthread_t t_arr[num_threads_ideal]; //thread array

    for(int e=0;e<epochs;e++)
    {
        int batches=train_images_file.images_num/batch_size;
        for(int b=0;b<batches;b++)
        {
            for(int thread=0;thread<num_threads_ideal;thread++)
            {
                if(thread==num_threads_ideal-1)
                {
                    buffer_arr[thread].img_per_state=batch_size/num_threads_ideal+batch_size%num_threads_ideal; //give the remaining images to last thread
                }
                else
                {
                    buffer_arr[thread].img_per_state=batch_size/num_threads_ideal;
                }
                pthread_create(&t_arr[thread],NULL,feedforward_and_backprop,&buffer_arr[thread]);
            }
            for(int thread=0;thread<num_threads_ideal;thread++)
            {
                pthread_join(t_arr[thread],NULL);
            }
        }
    }

    clear_state_arr(buffer_arr, num_threads_ideal);
}