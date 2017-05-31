#include "data_transform.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>

DataBlob* CenterCrop(DataBlob *bottom, uint crop_h, uint crop_w){
    uint n=0, c=0, h=0, w=0;

    DataBlob *top = (DataBlob*)MemoryPool(sizeof(DataBlob));
    top->n = bottom->n;
    top->c = bottom->c;
    top->h = crop_h;
    top->w = crop_w;
    top->data = (D_Type*)MemoryPool(sizeof(D_Type)*top->n*top->c*top->h*top->w);

    uint top_index=0, bottom_index=0;
    uint top_offset=0, bottom_offset=0;
    uint hstart = (uint)(ceil((float)(bottom->h-crop_h)/2));
    uint wstart = (uint)(ceil((float)(bottom->w-crop_w)/2));
    uint hend = hstart+crop_h;
    uint wend = wstart+crop_w;

    for (n=0;n<bottom->n;n=n+1){
        for (c=0;c<bottom->c;c=c+1){
            top_offset = n*top->c*top->h*top->w + c*top->h*top->c;
            bottom_offset = n*bottom->c*bottom->h*bottom->w+c*bottom->h*bottom->w;
            for (h=hstart;h<hend;h=h+1){
                for (w=wstart;w<wend;w=w+1){
                    top_index = top_offset+(h-hstart)*top->w+w-wstart;
                    bottom_index = bottom_offset+h*bottom->w+w;
                    top->data[top_index] = bottom->data[bottom_index];
                }
            }
        }
    }
    MemoryFree(bottom->data);
    MemoryFree(bottom);
    return top;
}

DataBlob* DataNormalize(DataBlob *bottom, const D_Type *mean, const D_Type *std_dev){
    uint n=0, c=0, h=0, w=0;

    DataBlob *top = bottom;
    for (n=0;n<bottom->n;n=n+1){
        for (c=0;c<top->c;c=c+1){
            uint top_offset = n*top->c*top->h*top->w+c*top->h*top->w;
            for (h=0;h<top->h;h=h+1){
                for (w=0;w<top->w;w=w+1){
                    uint top_index = top_offset+h*top->w+w;
                    top->data[top_index] = (bottom->data[top_index]-mean[c])/std_dev[c];
                }
            }
        }
    }
    return top;
}

void CenterCropTest(){
    D_Type input[25] = {
        1, -2, 3, 4, 5,
        -3, 4 , 5, 6, 1,
        -2, 3, 4, 5, -3,
        4 , 5, 6, 1, -2,
        3, 4, 5, -3, 4 };

    DataBlob *bottom = (DataBlob *)MemoryPool(sizeof(DataBlob));
    bottom->n = 1;
    bottom->c = 1;
    bottom->h = 5;
    bottom->w = 5;
    bottom->data = input;

    DataBlob *top = CenterCrop(bottom, 3, 3);
    PrintAll(top);
    printf("Test CenterCrop Pass\n");

}

void DataNormalizeTest(){
    D_Type input[25] = {
        1, -2, 3, 4, 5,
        -3, 4 , 5, 6, 1,
        -2, 3, 4, 5, -3,
        4 , 5, 6, 1, -2,
        3, 4, 5, -3, 4 };

    DataBlob *bottom = (DataBlob *)MemoryPool(sizeof(DataBlob));
    bottom->n = 1;
    bottom->c = 1;
    bottom->h = 5;
    bottom->w = 5;
    bottom->data = input;

    D_Type mean[1] = {2.32};
    D_Type std_dev[1] = {3.02};
    DataBlob *top = DataNormalize(bottom, mean, std_dev);
    PrintAll(top);
    printf("Test DataNormalize Pass\n");
}

