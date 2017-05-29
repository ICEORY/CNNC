#ifndef _RELU_H_
#define _RELU_H_
#include <math.h>
#include <malloc.h>
#include "utils.h"
#include "string.h"

/**
ReLU: y = max(x,0);
input: DataBlob *bottom, DataBlob *top
*/

void ReLU(DataBlob *bottom, DataBlob *top);

/**
test relu layer
state: pass
*/
void ReLUTest();

#endif // _RELU_H_
