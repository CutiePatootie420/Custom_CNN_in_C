#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NONE -1
#define SIZE 6
typedef struct temp
{
    int info;
    int count;
    int in_degree;
    struct temp** arr;
    

}temp;
typedef struct queue
{
    int size;
    struct temp** arr;
    int front;
    int rear;
}queue;
queue* init_queue(int size)
{
    queue* temp=malloc(sizeof(queue));
    temp->size=size;
    temp->arr=malloc(sizeof(struct temp*)*size);
    temp->front=0;
    temp->rear=0;
    return temp;
}
void enqueue(queue* q, temp* x)
{
    q->arr[q->rear]=x;
    q->rear=(q->rear+1)%q->size;
}
temp* dequeue(queue* q)
{
    temp* x=q->arr[q->front];
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
int main()
{
    temp* graph=malloc(sizeof(temp)*SIZE);
    int sizes[]={3,1,0,2,0,0};
    for(int i=0;i<SIZE;i++)
    {
        graph[i].count=sizes[i];
        graph[i].in_degree=0;
        graph[i].arr=malloc(sizeof(temp*)*graph[i].count);
        graph[i].info=i+1;
    }
    graph[0].arr[0]=&graph[1];
    graph[0].arr[1]=&graph[3];
    graph[0].arr[2]=&graph[4];
    graph[1].arr[0]=&graph[2];
    graph[3].arr[0]=&graph[2];
    graph[3].arr[1]=&graph[5];
    for(int i=0;i<SIZE;i++)
    {
        for(int j=0;j<graph[i].count;j++)
        {
            graph[i].arr[j]->in_degree++;
        }
    }

    temp** sorted_arr=malloc(sizeof(temp*)*SIZE);
    int count=0;
    queue* q=init_queue(SIZE+1);
    for(int i=0;i<SIZE;i++)
    {
        if(graph[i].in_degree==0)
        {
            enqueue(q,&graph[i]);
        }
    }
    int x;
    temp* curr;
    while(!is_empty(q))
    {
        curr=dequeue(q);
        sorted_arr[count++]=curr;
        for(int i=0;i<curr->count;i++)
        {
            x=--(curr->arr[i]->in_degree);
            if(x==0)
            {
                enqueue(q,curr->arr[i]);
            }
        }
    }
    for(int i=0;i<SIZE;i++)
    {
        printf("%c\n",sorted_arr[i]->info-1+'A');
    }
    free(sorted_arr);
    free(q->arr);
    free(q);
    for(int i=0;i<SIZE;i++)
    {
        free(graph[i].arr);
    }
    free(graph);
    return 0;
}


