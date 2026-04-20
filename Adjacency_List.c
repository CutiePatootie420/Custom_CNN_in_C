#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NONE -1
#define SIZE 6
typedef struct temp
{
    int info;
    int count;
    int in_order;
    struct temp** arr;
    

}temp;
int main()
{
    temp* graph=malloc(sizeof(temp)*SIZE);
    int sizes[]={3,1,0,2,0,0};
    for(int i=0;i<SIZE;i++)
    {
        graph[i].count=sizes[i];
        graph[i].in_order=0;
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
            graph[i].arr[j]->in_order++;
        }
    }
    for(int i=0;i<SIZE;i++)
    {
        printf("%c %d\n",i+'A',graph[i].in_order);
    }
    for(int i=0;i<SIZE;i++)
    {
        free(graph[i].arr);
    }
    free(graph);
    return 0;
}


