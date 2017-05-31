#ifndef _SCALE_H_
#define _SCALE_H_

#include "utils.h"

/**
scale layer
input:  DataBlob *bottom, DataBlob *top,
        const WeightBlob *gamma, const WeightBlob *beta, const uchar bias_term
return: top feature maps
*/
DataBlob* Scale(DataBlob *bottom,
           const WeightBlob *gamma, const WeightBlob *beta, const uchar bias_term);

/**
test scale layer
*/
void ScaleTest();

#endif // _SCALE_H_
