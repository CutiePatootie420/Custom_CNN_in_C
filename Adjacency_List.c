#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NONE -1
#define SIZE 6
typedef struct temp
{
    int info;
}temp;
int main()
{
    int** adjacency_mtx=malloc(sizeof(int*)*SIZE);
    adjacency_mtx[0]=malloc(sizeof(int)*4);
    adjacency_mtx[1]=malloc(sizeof(int)*2);
    adjacency_mtx[2]=malloc(sizeof(int)*1);
    adjacency_mtx[3]=malloc(sizeof(int)*3);
    adjacency_mtx[4]=malloc(sizeof(int)*1);
    adjacency_mtx[5]=malloc(sizeof(int)*1);
    adjacency_mtx[0][0]=1;
    adjacency_mtx[0][1]=3;
    adjacency_mtx[0][2]=4;
    adjacency_mtx[0][3]=NONE;
    adjacency_mtx[1][0]=2;
    adjacency_mtx[1][1]=NONE;
    adjacency_mtx[2][0]=NONE;
    adjacency_mtx[3][0]=2;
    adjacency_mtx[3][1]=5;
    adjacency_mtx[3][2]=NONE;
    adjacency_mtx[4][0]=NONE;
    adjacency_mtx[5][0]=NONE;
    temp* temp_arr=malloc(sizeof(temp)*SIZE);
    for(int i=0;i<SIZE;i++)
    {
        temp_arr[i].info=i+1;
    }
    int in_order[SIZE];
    memset(in_order,0,sizeof(in_order));
    for(int i=0;i<SIZE;i++)
    {
        for(int j=0;adjacency_mtx[i][j]!=NONE;j++)
        {
            in_order[adjacency_mtx[i][j]]++;
        }
    }
    for(int i=0;i<SIZE;i++)
    {
        printf("%c: %d\n",i+'A',in_order[i]);
    }


    for(int i=0;i<SIZE;i++)
    {
        free(adjacency_mtx[i]);
    }
    free(adjacency_mtx);
    free(temp_arr);
    return 0;
}


