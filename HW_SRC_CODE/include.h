#ifndef INCLUDE_H_
#define INCLUDE_H_
#include <iostream>
#include <stdlib.h>
#include<hls_half.h>
#include<assert.h>
#include<ap_fixed.h>
#include<ap_int.h>
#define dtype float

#define ftype  ap_fixed<24,9>
#define ftype_w ap_fixed<16,2>
//
//#define ftype ap_fixed<26,9>
//#define ftype_w ap_fixed<16,2>
////
//#define ftype_w float
//#define ftype float
//const int Tm=2;
//const int Tn=8;
//const int Tr=4;
//const int Tc=4;


const int Tm=20;
const int Tn=16;
const int Tr=20;
const int Tc=20;
//
const int R_K_C =1;
const int C_K_C =150;




//const int R_K_C =1;
//const int C_K_C =2;

void compute_conv(ftype input_fm_tile[Tn][(Tr-1)*1+R_K_C][(Tc-1)*1+C_K_C], ftype weights_tile[Tm][Tn][R_K_C][C_K_C], ftype output_fm_tile [Tm][Tr][Tc], ftype bias_tiled[Tm], int row, int col, int to, int ti, int RO,int CO,int M, int N, int S, int RK,int CK);
void perform_conv_tiled_main(float *input,float *output,int row,int col,int to,int ti, int N,int RO,int CO,float *weight, float *bias,int RK,int CK,int M,int S, int P,int RI,int CI,int row_in, int col_in);
#pragma SDS data zero_copy(input[0:14*299*256],output[0:14*299*256],weight[0:RK*CK*M*N], bias[0:M])
//#pragma SDS data access_pattern(input:SEQUENTIAL, output:SEQUENTIAL, weight:SEQUENTIAL,bias:SEQUENTIAL)
// #pragma SDS data zero_copy(bias[0:M])
// #pragma SDS data data_mover(input:AXIDMA_SIMPLE:1,output:AXIDMA_SIMPLE:2,weight:AXIDMA_SIMPLE:3)
//#pragma SDS data zero_copy(input[0:18],output[0:12],weight[0:RK*CK*M*N], bias[0:M],returns[0:100])
#pragma SDS data mem_attribute(weight:PHYSICAL_CONTIGUOUS, bias:PHYSICAL_CONTIGUOUS,input: PHYSICAL_CONTIGUOUS,output: PHYSICAL_CONTIGUOUS)
void perform_conv(float* input, float* output, float* weight, float* bias, int N, int M, int RI,int CI,int RO,int CO,int RK,int CK,int S, int P);
#pragma SDS data zero_copy(input[0:896],output[0:14],weight[0:896*14], bias[0:14])
#pragma SDS data mem_attribute(weight:PHYSICAL_CONTIGUOUS, bias:PHYSICAL_CONTIGUOUS,input: PHYSICAL_CONTIGUOUS,output: PHYSICAL_CONTIGUOUS)
void perform_dense (dtype *input, dtype *output, const dtype *weight, const dtype *bias,int Relu);
// void perform_conv(dtype* input,dtype* output,dtype* weight,dtype* bias,int M,int N,int Ri,int Ci,int Ro,int Co,int Rk,int Ck);



#endif
