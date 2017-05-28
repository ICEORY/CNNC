#ifndef _UTILS_H_
#define _UTILS_H_

#define uchar unsigned char
#define uint unsigned int
#define D_Type float

#define max(a,b) ((a>b)?a:b)
#define min(a,b) ((a<b)?a:b)

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


void PrintAll(DataBlob *data){
    uint data_count = data->n*data->c*data->h*data->w;
    uint i = 0;
    for (i=0;i<data_count;i=i+1){
        printf(">>> Data %d: %f\n",i, data->data[i]);
    }
}

#endif // _UTILS_H_
