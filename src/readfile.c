#include "readfile.h"
#include <stdio.h>
#include "utils.h"

uint ReadDatUInt(FILE *fp){
    uint f_input_data = 0;
    fscanf(fp, "%d ", &f_input_data);
    return f_input_data;
}

D_Type ReadDatDType(FILE *fp){
    D_Type f_input_data = 0;
    fscanf(fp, "%f ", &f_input_data);
    return f_input_data;
}

void ReadDatTest(){
    FILE *fp;
    fp = fopen("tools/network.dat", "r");
    uchar data;
    uint count = 0;
    while(!feof(fp)){
        switch (count){
            case 0:data = ReadDatUInt(fp);count=1;break;
            case 1:data = ReadDatDType(fp);count=0;break;
            default:count=0;break;
        }
        printf(">>>Read data is:%d\n", data);
    }
}


