#ifndef _READ_FILE_H_
#define _READ_FILE_H_
#include "utils.h"
#include <stdio.h>

/**
Read .dat file, return unsigned int
input: FILE *fp
*/
uint ReadDatUInt(FILE *fp);

/**
Read .dat file, return unsigned char
input: FILE *fp
*/
uchar ReadDatChar(FILE *fp);

/**
Read .dat file, return D_Type
input: FILE *fp
*/
D_Type ReadDatDType(FILE *fp);

/**
test ReadDatShort function
*/
void ReadDatTest();
#endif // _READ_FILE_H_
