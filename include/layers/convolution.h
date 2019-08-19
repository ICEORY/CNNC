#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include "utils.h"
/**
Convolutional layer
input:  DataBlob *bottom, DataBlob *top,
        const WeightBlob *weight, const WeightBlob *bias,
        const ParamsBlobS *params, const uchar bias_term
return: top feature maps
*/
DataBlob* Convolution(DataBlob *bottom,
                 const WeightBlob *weight, const WeightBlob *bias,
                 const ParamsBlobS *params, const uchar bias_term);


/**
Test Convolutional layer
*/
void ConvolutionTest();

#endif // _CONVOLUTION_H_
