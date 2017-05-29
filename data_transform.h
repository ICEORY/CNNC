#ifndef _DATA_TRANSFORM_H_
#define _DATA_TRANSFORM_H_

void CenterCrop(DataBlob *bottom, DataBlob *top, uint crop_h, uint crop_w){
    uint n=0, c=0, h=0, w=0;

    top->n = bottom->n;
    top->c = bottom->c;
    top->h = crop_h;
    top->w = crop_w;
    top->data = (D_Type*)MemoryPool(sizeof(D_Type)*top->n*top->c*top->h*top->w);

    uint top_index=0, bottom_index=0;
    uint top_offset=0, bottom_offset=0;
    uint hstart = (uint)(ceil((float)(bottom->h-crop_h)/2));
    uint wstart = (uint)(ceil((float)(bottom->w-crop_w)/2));

    for (n=0;n<bottom->n;n=n+1){
        for (c=0;c<bottom->c;c=c+1){
            top_offset = n*top->c*top->h*top->w + c*top->h*top->c;
            bottom_offset = n*bottom->c*bottom->h*bottom->w+c*bottom->h*bottom->w;
            for (h=hstart;h<crop_h;h=h+1){
                for (w=wstart;w<crop_w;w=w+1){
                    top_index = top_offset+(h-hstart)*top->w+w-wstart;
                    bottom_index = bottom_offset+h*bottom->w+w;
                    top->data[top_index] = bottom->data[bottom_index];
                }
            }
        }
    }
    MemoryFree(bottom);
}

void DataNormalize(DataBlob *bottom, DataBlob *top, const D_Type *mean, const D_Type *std_dev){
    uint n=0, c=0, h=0, w=0;

    top = bottom;
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
}
#endif // _DATA_TRANSFORM_H_
