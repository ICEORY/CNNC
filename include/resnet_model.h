#ifndef _RESNET_MODEL_H_
#define _RESNET_MODEL_H_
#include "utils.h"
/*
void ReadWeight(char *file_path, D_Type *data, uint weight_count){
    FILE *fp;
    fp = fopen(file_path, "r");
    fread(data, sizeof(D_Type)*weight_count, weight_count, fp);
}*/

/**
wrapper of convolutional layer, kernel_size is 3x3
*/
void conv3x3_layer(DataBlob *bottom, DataBlob *top, uint in_plane, uint out_plane, uchar stride, D_Type *weight_data);

/**
wrapper of convolutional layer, kernel_size is 1x1
*/
void conv1x1_layer(DataBlob *bottom, DataBlob *top, uint in_plane, uint out_plane, uchar stride, D_Type *weight_data);

/**
wrapper of batch_norm_layer layer
*/
void batch_norm_layer(DataBlob *bottom, DataBlob *top, const D_Type* weight_data);

/**
wrapper of linear layer
*/
void linear_layer (DataBlob *bottom, DataBlob *top, D_Type* weight_data);

/**
wrapper of residual branch
*/
void ResidualBranch(DataBlob *bottom, DataBlob *top,
                    uint in_plane, uint out_plane, uchar stride,
                    D_Type **weight);

/**
define resnet 20
*/
void ResNet_20(DataBlob *bottom, DataBlob *top, D_Type **weight);

#endif // _RESNET_MODEL_H_
