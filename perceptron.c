#include <stdio.h>
#include <stdlib.h>
#define BIT_ADDER_SIZE 7
typedef struct perceptron
{
    int info;
    double bias;
    int in_degree;
    int in_degree_copy;
    int next_count;
    struct perceptron** next_arr;
    double* weights;

}perceptron;
typedef struct queue
{
    int size;
    struct perceptron** arr;
    int front;
    int rear;
}queue;
queue* init_queue(int size)
{
    queue* temp=malloc(sizeof(queue));
    temp->size=size;
    temp->arr=malloc(sizeof(struct perceptron*)*size);
    temp->front=0;
    temp->rear=0;
    return temp;
}
void enqueue(queue* q, perceptron* x)
{
    q->arr[q->rear]=x;
    q->rear=(q->rear+1)%q->size;
}
perceptron* dequeue(queue* q)
{
    perceptron* x=q->arr[q->front];
    q->front=(q->front+1)%q->size;
    return x;
}
int is_empty(queue* q)
{
    if(q->rear==q->front)
    {
        return 1;
    }
    return 0;
}
void compute_output(perceptron* p)
{
    p->info=((p->info+p->bias)<=0)?0:1;
}
void fire_perceptron(perceptron* p)
{
    for(int i=0;i<p->next_count;i++)
    {
        p->next_arr[i]->info+=p->info*p->weights[i];
    }
}
int main()
{
    perceptron* bit_adder_graph=malloc(sizeof(perceptron)*BIT_ADDER_SIZE);
    int sizes[]={2,2,4,1,1,0,0};
    for(int i=0;i<BIT_ADDER_SIZE;i++)
    {
        bit_adder_graph[i].info=0;
        bit_adder_graph[i].next_count=sizes[i];
        bit_adder_graph[i].next_arr=malloc(sizeof(perceptron*)*bit_adder_graph[i].next_count);
        bit_adder_graph[i].weights=malloc(sizeof(double)*bit_adder_graph[i].next_count);
        bit_adder_graph[i].bias=3;
        bit_adder_graph[i].in_degree=0;
        bit_adder_graph[i].in_degree_copy=0;
        for(int j=0;j<bit_adder_graph[i].next_count;j++)
        {
            bit_adder_graph[i].weights[j]=-2;
        }
    }

    bit_adder_graph[0].next_arr[0]=&bit_adder_graph[2];
    bit_adder_graph[0].next_arr[1]=&bit_adder_graph[3];
    bit_adder_graph[1].next_arr[0]=&bit_adder_graph[2];
    bit_adder_graph[1].next_arr[1]=&bit_adder_graph[4];
    bit_adder_graph[2].next_arr[0]=&bit_adder_graph[3];
    bit_adder_graph[2].next_arr[1]=&bit_adder_graph[4];
    bit_adder_graph[2].next_arr[2]=&bit_adder_graph[6];
    bit_adder_graph[2].next_arr[3]=&bit_adder_graph[6];
    bit_adder_graph[3].next_arr[0]=&bit_adder_graph[5];
    bit_adder_graph[4].next_arr[0]=&bit_adder_graph[5];

    for(int i=0;i<BIT_ADDER_SIZE;i++)
    {
        for(int j=0;j<bit_adder_graph[i].next_count;j++)
        {
            bit_adder_graph[i].next_arr[j]->in_degree++;
            bit_adder_graph[i].next_arr[j]->in_degree_copy++;
        }
    }

    perceptron** sorted_arr=malloc(sizeof(perceptron*)*BIT_ADDER_SIZE);
    int index=0;
    queue* q=init_queue(BIT_ADDER_SIZE+1);    
    for(int i=0;i<BIT_ADDER_SIZE;i++)
    {
        if(bit_adder_graph[i].in_degree==0)
        {
            enqueue(q,&bit_adder_graph[i]);
        }
    }
    int x;
    perceptron* curr;
    while(!is_empty(q))
    {
        curr=dequeue(q);
        sorted_arr[index++]=curr;
        for(int i=0;i<curr->next_count;i++)
        {
            x=--(curr->next_arr[i]->in_degree);
            if(x==0)
            {
                enqueue(q,curr->next_arr[i]);
            }
        }
    }
    bit_adder_graph[0].info=0; //set input1
    bit_adder_graph[1].info=0; //set input1

    for(int i=0;i<index;i++)
    {
        if(sorted_arr[i]->in_degree_copy!=0)
        {
            compute_output(sorted_arr[i]);
        }
        fire_perceptron(sorted_arr[i]);
    }

    for(int i=0;i<index;i++) //check which index is giving sum and carry
    {
        printf("%ld ", sorted_arr[i] - bit_adder_graph);
    }
    printf("\nsum: %d, carry: %d",sorted_arr[6]->info,sorted_arr[5]->info);

    free(sorted_arr);
    free(q->arr);
    free(q);
    for(int i=0;i<BIT_ADDER_SIZE;i++)
    {
        free(bit_adder_graph[i].next_arr);
        free(bit_adder_graph[i].weights);
    }
    free(bit_adder_graph);
}