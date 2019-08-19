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
DataBlob* conv3x3_layer(DataBlob *bottom, uint in_plane, uint out_plane, uchar stride, D_Type *weight_data);

/**
wrapper of convolutional layer, kernel_size is 1x1
*/
DataBlob* conv1x1_layer(DataBlob *bottom, uint in_plane, uint out_plane, uchar stride, D_Type *weight_data);

/**
wrapper of batch_norm_layer layer
*/
DataBlob* batch_norm_layer(DataBlob *bottom, D_Type* weight_data);

/**
wrapper of scale layer
*/
DataBlob* scale_layer(DataBlob *bottom, D_Type* weight_data);

/**
wrapper of linear layer
*/
DataBlob* linear_layer (DataBlob *bottom, D_Type* weight_data);

/**
wrapper of avg pooling layer
*/
DataBlob* avg_pooling_layer(DataBlob *bottom);

//==========================
/**
wrapper of Ensemble_conv(bias)_bn_(non beta)_relu
*/
DataBlob* convb_bnn_relu(DataBlob *bottom, uint in_plane, uint out_plane,
                         uchar kernel, uchar padding, uchar stride,
                         D_Type *conv_weight, D_Type *bn_weight, D_Type *scale_weight);

/**
wrapper of ensemble_conv(bias)_bn(non beta)_(non relu)
*/
DataBlob* convb_bnn(DataBlob *bottom, uint in_plane, uint out_plane,
                    uchar kernel, uchar padding, uchar stride,
                    D_Type *conv_weight, D_Type *bn_weight, D_Type *scale_weight);
//==========================
/**
wrapper of residual branch
*/
DataBlob* ResidualBranch(DataBlob *bottom,
                    uint in_plane, uint out_plane, uchar stride,
                    D_Type **weight, uchar ptr_offset);

/**
wrapper of residual block
*/
DataBlob* ResidualBlock(DataBlob *bottom, uint in_plane, uint out_plane, D_Type **weight, uchar down_sample_term, uchar ptr_offset);
/**
define resnet 20
*/
DataBlob* ResNet_20(DataBlob *bottom, D_Type **weight);

/**
test resnet 20
*/
void resnet20test();
#endif // _RESNET_MODEL_H_
