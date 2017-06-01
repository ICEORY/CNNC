#ifndef _CONV_BN_SCALE_RELU_H_
#define _CONV_BN_SCALE_RELU_H_
#include "utils.h"

/**
combine convolution, batch normalization, scale and relu together
input:  bottom feature maps, parameters: kernel_size, padding_size, stride,
        in_plane and out_plane for conv
        weights: for conv, bn and scale
conv here has bias and scale layer without beta,
name of this function: conv(bias)_bn(non beta)_relu
output: top feature maps;
*/
DataBlob* Ensemble_Convb_BNn_ReLU(DataBlob* bottom, ParamsBlobL *params,
                                  uint in_plane, uint out_plane,
                                  D_Type *conv_weight, D_Type *conv_bias,
                                  D_Type *bn_mean, D_Type *bn_var, D_Type bn_scale_factor,
                                  D_Type *scale_gamma);

/**
combine convolution, batch normm, scale together, without relu
input:  bottom feature maps, parameters: kernel_size, padding_size, stride,
        in_plane and out_plane for conv
        weights: for conv, bn and scale
output: top feature maps
*/
DataBlob* Ensemble_Convb_BNn(DataBlob* bottom, ParamsBlobL *params,
                             uint in_plane, uint out_plane,
                             D_Type *conv_weight, D_Type *conv_bias,
                             D_Type *bn_mean, D_Type *bn_var, D_Type bn_scale_factor,
                             D_Type *scale_gamma);
/**
you can define more function by yourself, for example: ensemble conv without bias and batch norm with beta, etc.
*/
#endif // _CONV_BN_SCALE_RELU_H_
