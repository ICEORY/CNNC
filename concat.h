#ifndef _CONCAT_H_
#define _CONCAT_H_

#include "utils.h"

void ConcatTable(DataBlob *bottom, DataBlob *top_1, DataBlob *top_2){
    uint top_count = bottom->n * bottom->c * bottom->h * bottom->w;
    uint i =0;

    top_1 = bottom;

    top_2->n = bottom->n;
    top_2->c = bottom->c;
    top_2->h = bottom->h;
    top_2->w = bottom->w;
    top_2->data = (D_Type*)MemoryPool(sizeof(D_Type)*bottom->n*bottom->c*bottom->h*bottom->w);

    for (i=0;i<top_count;i=i+1){
        top_2->data[i] = bottom->data[i];
    }

}

void CAddTable(DataBlob *bottom_1, DataBlob *bottom_2, DataBlob *top){
    uint top_count = bottom_1->n * bottom_1->c * bottom_1->h * bottom_1->w;
    uint i =0;

    top = bottom_1;
    for (i=0;i<top_count;i=i+1){
        top->data[i] = bottom_1->data[i]+bottom_2->data[i];
    }
    MemoryFree(bottom_2);
}
#endif // _CONCAT_H_
