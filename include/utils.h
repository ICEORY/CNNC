/** \project Convolutional Neural Networks in C: CNN-C
 * \author iceory
 * \emal z.zhuangwei@scut.edu.cn
 * \date 2017.5.29
 * \reference: https://github.com/BVLC/caffe
 */
#ifndef _UTILS_H_
#define _UTILS_H_


#include <malloc.h>
#include "string.h"

#define uchar unsigned char
#define uint unsigned int
#define D_Type float

#define max(a,b) ((a>b)?a:b)
#define min(a,b) ((a<b)?a:b)

#define BN_EPS 0.00001
//#define NULL (void*)0

/**
define blob for save input or output data
parameters:
n: batch size / in_plane
c: channel / out_plane
h: height / kernel_size_h
w: width / kernel_size_w
*data: memory for saving data
*/

typedef struct DataBlob_{
    uint n;
    uint c;
    uint h;
    uint w;
    D_Type *data;
}DataBlob;


/**
define blob for save weight data
parameters:
in_plane
out_plane
kernel_h
kernel_w
*data: memory for saving data
*/
typedef struct WeightBlob_{
    uint in_plane;
    uint out_plane;
    uchar kernel_h;
    uchar kernel_w;
    D_Type *data;
}WeightBlob;


/**
define blob to describe optional parameters
parameters:
padding_h
padding_w
stride_h
stride_w
*/
typedef struct ParamsBlobSmall_{
    uchar padding_h;
    uchar padding_w;
    uchar stride_h;
    uchar stride_w;
}ParamsBlobS;

/**
define blob to describe optional parameters
parameters:
kernel_h
kernel_w
padding_h
padding_w
stride_h
stride_w
*/
typedef struct ParamsBlobLarge_{
    uchar kernel_h;
    uchar kernel_w;
    uchar padding_h;
    uchar padding_w;
    uchar stride_h;
    uchar stride_w;
}ParamsBlobL;

/**-----------------------------------------------------------------*/
/**
print all data of DataBlob object
input: target DataBlob data
*/
void PrintAll(DataBlob*);

/**
TO DO:
you need to re-write these two function for more efficient management of memory pool
*/
/**
assign memory
input: unsigned int memory_size
*/
void* MemoryPool(long);

/**
free memory
input: void *ptr
*/
void MemoryFree(void*);

#endif // _UTILS_H_
