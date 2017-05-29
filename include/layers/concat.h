#ifndef _CONCAT_H_
#define _CONCAT_H_

#include "utils.h"

/**
ConcatTable Layer:
input:  DataBlob *bottom, DataBlob *top_1, DataBlob *top_2
*/
void ConcatTable(DataBlob *bottom, DataBlob *top_1, DataBlob *top_2);

/**
CAddTable Layer
input:  DataBlob *bottom_1, DataBlob *bottom_2, DataBlob *top
*/
void CAddTable(DataBlob *bottom_1, DataBlob *bottom_2, DataBlob *top);

/**
Test ConcatTable Layer
*/
void ConcatTableTest();

/**
Test CAddTable Layer
*/
void CAddTableTest();
#endif // _CONCAT_H_
