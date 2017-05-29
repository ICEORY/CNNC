#ifndef _AVG_POOLING_H_
#define _AVG_POOLING_H_

#include "utils.h"
/**
average pooling layer
input: DataBlob *bottom, Data *top, ParamsBlobL *params
*/
void AvgPooling(DataBlob *bottom, DataBlob *top, const ParamsBlobL *params);

/**
test avg_pooling_layer
*/
void AvgPoolingTest();
#endif // _AVG_POOLING_H_
