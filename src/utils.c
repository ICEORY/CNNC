#include "utils.h"
#include <stdio.h>
#include <malloc.h>
#include "string.h"
/**
print all data of DataBlob object
*/
void PrintAll(DataBlob *data){
    uint data_index = 0;
    uint n=0, c=0, h=0, w=0;
    printf("====================================\n");
    for (n=0;n<data->n;n=n+1){
        for (c=0;c<data->c;c=c+1){
            printf(">>>Data (%d, %d, :, :,)\n", n, c);
            for (h=0;h<data->h;h=h+1){
                for (w=0;w<data->w;w=w+1){
                    data_index = n*data->c*data->h*data->w+c*data->h*data->w+h*data->w+w;
                    printf("%f\t", data->data[data_index]);
                }
                printf("\n");
            }
        }
    }
    /*for (i=0;i<data_count;i=i+1){
        printf(">>> Data %d: %f\n",i, data->data[i]);
    }*/
    printf("------------------------------------\n");
}

/**
TO DO:
you need to re-write these two function for more efficient management of memory pool
*/
void* MemoryPool(long mem_size){
    void *ptr = (void *) malloc(mem_size);
    memset(ptr, 0, mem_size);
    return ptr;
}

void MemoryFree(void *p){
    free(p);
}
