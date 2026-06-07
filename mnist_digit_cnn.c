#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <time.h>
typedef struct layer
{
    double* activation;
    double* biases;
    double** weights;
}layer;
typedef struct vector
{
    double* parameter;
}vector;
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
double sigmoid_func(double x)
{
    return 1.0/(1.0+exp(-x));
}
double random_weight()
{
    return (((double)rand()/RAND_MAX)-0.5); 
}
void fill_image_indices(int size, int* arr,int num)
{
    for(int i=0;i<size;i++)
    {
        arr[i]=abs(rand())%num;
    }
}
void load_image(layer* cnn,unsigned char* pixels, int index, int size)
{
    for(int pixel=0;pixel<size;pixel++)
    {
        cnn[0].activation[pixel]=pixels[index*size+pixel]/255.0;
    }
}
double sigmoid_derivative(double x)
{
    double temp=exp(x);
    return (temp/((1.0 + temp)*(1.0 + temp)));
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

    int cnn_summary[]={image_size, 16,16,10}; //building cnn
    int num_layers=sizeof(cnn_summary)/sizeof(cnn_summary[0]);
    layer* cnn=malloc(sizeof(layer)*num_layers);
    for(int layer=0;layer<num_layers;layer++)
    {
        if(layer==0)
        {
            cnn[layer].biases=NULL;
        }
        else
        {
            cnn[layer].biases=malloc(sizeof(double)*cnn_summary[layer]);
        }
        cnn[layer].activation=malloc(sizeof(double)*cnn_summary[layer]);
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

    for(int layer=0;layer<num_layers;layer++) //setting parameters to random
    {
        for(int node=0;node<cnn_summary[layer];node++)
        {
            cnn[layer].activation[node]=random_weight();
            if(layer!=0)
            {
                cnn[layer].biases[node]=random_weight();
            }
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

    vector cost;
    vector* Z=malloc(sizeof(vector)*(num_layers)); 
    vector* dc_da=malloc(sizeof(vector)*(num_layers)); // dc/da
    vector* dc_dw=malloc(sizeof(vector)*(num_layers)); // dc/dw
    vector* dc_db=malloc(sizeof(vector)*(num_layers)); // dc/db
    cost.parameter=malloc(sizeof(double)*cnn_summary[num_layers-1]);
    for(int i=0;i<num_layers;i++)
    {
        if(i==0)
        {
            dc_da[i].parameter=NULL;
            continue;
        }
        dc_da[i].parameter=malloc(sizeof(double)*cnn_summary[i]);
    }
    for(int i=0;i<num_layers;i++)
    {
        if(i==num_layers-1)
        {
            dc_dw[i].parameter=NULL;
            continue;
        }
        dc_dw[i].parameter=malloc(sizeof(double)*cnn_summary[i]*cnn_summary[i+1]);
    }
    for(int i=0;i<num_layers;i++)
    {
        if(i==0)
        {
            dc_db[i].parameter=NULL;
            continue;
        }
        dc_db[i].parameter=malloc(sizeof(double)*cnn_summary[i]);
    }
    for(int i=0;i<num_layers;i++)
    {
        if(i==0)
        {
            Z[i].parameter=NULL;
            continue;
        }
        Z[i].parameter=malloc(sizeof(double)*cnn_summary[i]);
    }
    
    int epochs=5;
    int batch_size=32;
    int image_indices[batch_size];
    
    
    double learning_rate=0.1;

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for(int e=0;e<epochs;e++) //training 
    {
        for(int mini_batch=0;mini_batch<images_num/batch_size;mini_batch++)
        {
            for(int i=0;i<num_layers;i++) //setting derivative vectors to 0
            {
                if(i==0)
                {
                    continue;
                }
                memset(dc_da[i].parameter,0,sizeof(double)*cnn_summary[i]);
            }
            for(int i=0;i<num_layers;i++)
            {
                if(i==num_layers-1)
                {
                    continue;
                }
                memset(dc_dw[i].parameter,0,sizeof(double)*cnn_summary[i]*cnn_summary[i+1]);
            }
            for(int i=0;i<num_layers;i++)
            {
                if(i==0)
                {
                    continue;
                }
                memset(dc_db[i].parameter,0,sizeof(double)*cnn_summary[i]);
            }
                
            fill_image_indices(batch_size,image_indices,images_num); //determine batch
            for(int image=0;image<batch_size;image++) 
            {
                load_image(cnn,pixels,image_indices[image],image_size);

                double temp_final_layer=0;
                for(int i=0;i<num_layers-1;i++) //processing cnn layers (feedforward)
                {
                    int j=i+1;
                    double temp;
                    for(int next_node=0;next_node<cnn_summary[j];next_node++)
                    {
                        temp=0;
                        for(int node=0;node<cnn_summary[i];node++)
                        {
                            temp+=cnn[i].activation[node]*cnn[i].weights[node][next_node];
                        }
                        temp+=cnn[j].biases[next_node];
                        Z[j].parameter[next_node]=temp;
                        if(j!=num_layers-1)
                        {
                            cnn[j].activation[next_node]=sigmoid_func(temp);
                        }
                        else 
                        {
                            cnn[j].activation[next_node]=exp(temp);
                            temp_final_layer+=cnn[j].activation[next_node];
                        }
                    }
                }
                for(int i=0;i<cnn_summary[num_layers-1];i++)
                {
                    cnn[num_layers-1].activation[i]/=temp_final_layer;
                }
                
                for(int i=0;i<cnn_summary[num_layers-1];i++) //computing cost 
                {
                    if(labels[image_indices[image]]==i)
                    {
                        cost.parameter[i]=(cnn[num_layers-1].activation[i]-1);
                    }
                    else
                    {
                        cost.parameter[i]=(cnn[num_layers-1].activation[i]);
                    }
                }

                int curr_layer=num_layers-1; //calculating derivative vectors 
                for(int node=0;node<cnn_summary[curr_layer];node++) //for last layer (l3)
                {
                    //dc_da[curr_layer].parameter[node]+=cost.parameter[node]/(cnn[curr_layer].activation[node]*(1-cnn[curr_layer].activation[node]));
                    dc_db[curr_layer].parameter[node]+=cost.parameter[node];
                    for(int prev_node=0;prev_node<cnn_summary[curr_layer-1];prev_node++) //calculating dc/dw for l2
                    {
                        dc_dw[curr_layer-1].parameter[prev_node*cnn_summary[num_layers-1]+node]+=cost.parameter[node]*cnn[curr_layer-1].activation[prev_node];
                    }
                }
                curr_layer--; //2
                for(int node=0;node<cnn_summary[curr_layer];node++) //l2
                {
                    for(int node_final_layer=0;node_final_layer<cnn_summary[num_layers-1];node_final_layer++) 
                    {
                        dc_da[curr_layer].parameter[node]+=cost.parameter[node_final_layer]*cnn[curr_layer].weights[node][node_final_layer];
                        dc_db[curr_layer].parameter[node]+=cost.parameter[node_final_layer]*cnn[curr_layer].weights[node][node_final_layer]*sigmoid_derivative(Z[curr_layer].parameter[node]);
                        for(int prev_node=0;prev_node<cnn_summary[curr_layer-1];prev_node++) //calculating dc/dw for l1
                        {
                            dc_dw[curr_layer-1].parameter[prev_node*cnn_summary[curr_layer]+node]+=cost.parameter[node_final_layer]*cnn[curr_layer].weights[node][node_final_layer]*sigmoid_derivative(Z[curr_layer].parameter[node])*cnn[curr_layer-1].activation[prev_node];
                        }
                    }
                }
                curr_layer--; //1
                for(int node=0;node<cnn_summary[curr_layer];node++) //l1
                {
                    for(int node_final_layer=0;node_final_layer<cnn_summary[num_layers-1];node_final_layer++)
                    {
                        double temp=0;
                        for(int node_next=0;node_next<cnn_summary[curr_layer+1];node_next++)
                        {
                            temp+=cnn[curr_layer+1].weights[node_next][node_final_layer]*sigmoid_derivative(Z[curr_layer+1].parameter[node_next])*cnn[curr_layer].weights[node][node_next];
                        }
                        dc_da[curr_layer].parameter[node]+=cost.parameter[node_final_layer]*temp;
                        dc_db[curr_layer].parameter[node]+=cost.parameter[node_final_layer]*temp*sigmoid_derivative(Z[curr_layer].parameter[node]);
                        for(int prev_node=0;prev_node<cnn_summary[curr_layer-1];prev_node++) //calculating dc/dw for l0
                        {
                            dc_dw[curr_layer-1].parameter[prev_node*cnn_summary[curr_layer]+node]+=cost.parameter[node_final_layer]*temp*sigmoid_derivative(Z[curr_layer].parameter[node])*cnn[curr_layer-1].activation[prev_node];
                        }
                    }
                } 
                /*if(mini_batch%500==0 && image==31)
                {
                    printf("Output layer , epoch %d, mini-batch %d, iteration %d: \n",e,mini_batch,image); //Printing final layer activations and prediction
                    int prediction=0;
                    for(int i=0;i<cnn_summary[num_layers-1];i++)
                    {
                        printf("%d: %f\n",i,cnn[num_layers-1].activation[i]);
                        if(cnn[num_layers-1].activation[i]>cnn[num_layers-1].activation[prediction])
                        {
                            prediction=i;
                        }
                    }
                    printf("Prediction: %d\n Actual: %d\n",prediction,labels[image_indices[image]]);
                }*/ 
            }
            for(int layer=0;layer<num_layers;layer++) //X=x-learning_rate*derivative/batch_size
            {
                if(layer!=0)
                {
                    for(int node=0;node<cnn_summary[layer];node++) //changing biases
                    {
                        cnn[layer].biases[node]-=(learning_rate*dc_db[layer].parameter[node]/batch_size);
                    }
                }
                if(layer!=num_layers-1)
                {
                    for(int node=0;node<cnn_summary[layer];node++)
                    {
                        for(int node_next_layer=0;node_next_layer<cnn_summary[layer+1];node_next_layer++)
                        {
                            cnn[layer].weights[node][node_next_layer]-=(learning_rate*dc_dw[layer].parameter[node*cnn_summary[layer+1]+node_next_layer]/batch_size);
                        }
                    }
                }                
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC,&end_time);
    double elapsed_time=(end_time.tv_sec-start_time.tv_sec)+(end_time.tv_nsec-start_time.tv_nsec)/1e9;
    printf("Training Time (Old Monolithic): %f seconds\n",elapsed_time);

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
    double accuracy=0;

    for(int index=0;index<test_images_num;index++) //testing
    {
        load_image(cnn,test_pixels,index, image_size);

        double temp_final_layer=0;
        for(int i=0;i<num_layers-1;i++)
        {
            int j=i+1;
            double temp;
            for(int next_node=0;next_node<cnn_summary[j];next_node++)
            {
                temp=0;
                for(int node=0;node<cnn_summary[i];node++)
                {
                    temp+=cnn[i].activation[node]*cnn[i].weights[node][next_node];
                }
                temp+=cnn[j].biases[next_node];
                Z[j].parameter[next_node]=temp;
                if(j!=num_layers-1)
                {
                    cnn[j].activation[next_node]=sigmoid_func(temp);
                }
                else 
                {
                    cnn[j].activation[next_node]=exp(temp);
                    temp_final_layer+=cnn[j].activation[next_node];
                }
            }
        }
        for(int i=0;i<cnn_summary[num_layers-1];i++)
        {
            cnn[num_layers-1].activation[i]/=temp_final_layer;
        }

        int prediction=0;
        for(int i=0;i<cnn_summary[num_layers-1];i++)
        {
            if(cnn[num_layers-1].activation[i]>cnn[num_layers-1].activation[prediction])
            {
                prediction=i;
            }
        }
        if(prediction==test_labels[index])
        {
            accuracy++;
        }
        if(index%100==0)
        {
            printf("Prediction: %d, actual: %d\n",prediction,test_labels[index]);
        }
        
    }
    accuracy/=test_images_num;
    printf("Accuracy over %d epochs, %d batch size on learning rate %.2f:   %f\n",epochs,batch_size,learning_rate,accuracy);


    free(test_pixels);
    free(test_labels);
    fclose(train_images_file);
    fclose(label_images_file);
    fclose(test_images_file);
    fclose(test_labels_file);
    free(pixels);
    free(labels);
    free(cost.parameter);
    for(int layer=0;layer<num_layers;layer++)
    {
        if(cnn[layer].biases!=NULL)
        {
            free(cnn[layer].biases);
        }
        if(cnn[layer].activation!=NULL)
        {
            free(cnn[layer].activation);
        }
        if(cnn[layer].weights!=NULL)
        {
            for(int i=0;i<cnn_summary[layer];i++)
            {
                free(cnn[layer].weights[i]);
            }
            free(cnn[layer].weights);
        }
    }
    free(cnn);
    for(int i=0; i<num_layers; i++)
    {
        if(Z[i].parameter != NULL)
        {
            free(Z[i].parameter);
        }
        if(dc_da[i].parameter != NULL)
        {
            free(dc_da[i].parameter);
        }
        if(dc_dw[i].parameter != NULL)
        {
            free(dc_dw[i].parameter);
        }
        if(dc_db[i].parameter != NULL)
        {
            free(dc_db[i].parameter);
        }
    }
    free(Z);
    free(dc_da);
    free(dc_dw);
    free(dc_db);
    return 0;
}
