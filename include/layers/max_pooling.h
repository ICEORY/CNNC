#ifndef _MAX_POOLING_H_
#define _MAX_POOLING_H_

#include "utils.h"

/**
Max Pooling Layer
input: DataBlob *bottom, DataBlob *top, const ParamsBlobL *params
return: top feature maps
*/
DataBlob* MaxPooling(DataBlob *bottom, const ParamsBlobL *params);

/**
test max pooling layer
*/
void MaxPoolingTest();

#endif // _MAX_POOLING_H_
