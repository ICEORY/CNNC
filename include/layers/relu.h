#ifndef _RELU_H_
#define _RELU_H_
#include <math.h>
#include <malloc.h>
#include "utils.h"
#include "string.h"

/**
ReLU: y = max(x,0);
input: DataBlob *bottom, DataBlob *top
return: top feature maps
*/

DataBlob* ReLU(DataBlob *bottom);

/**
test relu layer
state: pass
*/
void ReLUTest();

#endif // _RELU_H_
