#ifndef _CONCAT_H_
#define _CONCAT_H_

#include "utils.h"

/**
ConcatTable Layer:
input:  DataBlob *bottom,  DataBlob *top_2 (top_2 here is only a pointer, without initialized, and it will be modified on the function)
return: top_1 feature maps, top_2 feature maps are modified, too.
*/
DataBlob* ConcatTable(DataBlob *bottom, DataBlob *top_2);

/**
CAddTable Layer
input:  DataBlob *bottom_1, DataBlob *bottom_2, DataBlob *top
return: output feature maps
*/
DataBlob* CAddTable(DataBlob *bottom_1, DataBlob *bottom_2);

/**
Test ConcatTable Layer
*/
void ConcatTableTest();

/**
Test CAddTable Layer
*/
void CAddTableTest();
#endif // _CONCAT_H_
