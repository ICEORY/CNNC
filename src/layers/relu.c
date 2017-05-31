#include "relu.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>

/**
ReLU: y = max(x,0);
*/

DataBlob* ReLU(DataBlob *bottom){

    uint top_count = bottom->n * bottom->c * bottom->h * bottom->w;
    uint i =0;
    DataBlob *top = bottom;
    for (i=0;i<top_count;i=i+1){
        if (top->data[i]<0){
            top->data[i] = 0;
        }
    }
    //printf(">>>relu: n:%d, c:%d, h:%d, w:%d\n",top->n, top->c, top->h, top->w);
    return top;

}


/**
test relu layer
*/
void ReLUTest(){
    D_Type input[9] = {1, -2, 3, 4, 5, -3, 4 , 5, 6};
    DataBlob *bottom = (DataBlob *)MemoryPool(sizeof(DataBlob));
    bottom->n = 1;
    bottom->c = 1;
    bottom->h = 3;
    bottom->w = 3;
    bottom->data = input;

    DataBlob *top = ReLU(bottom);
    PrintAll(top);
    printf("ReLUTest Pass\n");
}
