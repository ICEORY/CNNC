#ifndef _LINEAR_H_
#define _LINEAR_H_

#include "utils.h"
/**
input dim: N_i*C_i*1*1
output dim: N_o*C_o*1*1
weight dim: N_w*C_i*C_o
where N_i = N_o = N_w = batch size

Linear layer:
input:  DataBlob *bottom, DataBlob *top,
        const WeightBlob *weight, const WeightBlob *bias,
        const uchar bias_term
output: top feature maps
*/
DataBlob* Linear(DataBlob *bottom,
            const WeightBlob *weight, const WeightBlob *bias,
            const uchar bias_term);

/**
test linear layer
*/
void LinearTest();
#endif // _LINEAR_H_
