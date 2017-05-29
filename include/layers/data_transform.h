#ifndef _DATA_TRANSFORM_H_
#define _DATA_TRANSFORM_H_

#include "utils.h"
/**
center crop:
input: DataBlob *bottom, DataBlob *top, uint crop_h, uint crop_w
*/
void CenterCrop(DataBlob *bottom, DataBlob *top, uint crop_h, uint crop_w);

/**
normalize:
input: DataBlob *bottom, DataBlob *top, const D_Type *mean, const D_Type *std_dev
*/
void DataNormalize(DataBlob *bottom, DataBlob *top, const D_Type *mean, const D_Type *std_dev);

/**
test center crop function
*/
void CenterCropTest();

/**
test data normalize function
*/
void DataNormalizeTest();
#endif // _DATA_TRANSFORM_H_
