#ifndef _BATCH_NORMALIZATION_H_
#define _BATCH_NORMALIZATION_H_

#include "utils.h"
/**
batch normalization layer
input:  DataBlob *bottom, DataBlob *top,
        const WeightBlob *mean, const WeightBlob *var,
        const WeightBlob *gamma, const WeightBlob *beta,
        D_Type scale_factor, const D_Type eps
*/
void BatchNormalization(DataBlob *bottom, DataBlob *top,
                        const WeightBlob *mean, const WeightBlob *var,
                        const WeightBlob *gamma, const WeightBlob *beta,
                        D_Type scale_factor, const D_Type eps);


/**
test batch normalization layer
*/
void BatchNormalizationTest();
#endif // _BATCH_NORMALIZATION_H_
