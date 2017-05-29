#ifndef _SCALE_H_
#define _SCALE_H_

#include "utils.h"

/**
scale layer
input:  DataBlob *bottom, DataBlob *top,
        const WeightBlob *gamma, const WeightBlob *beta
*/
void Scale(DataBlob *bottom, DataBlob *top,
           const WeightBlob *gamma, const WeightBlob *beta);

/**
test scale layer
*/
void ScaleTest();

#endif // _SCALE_H_
