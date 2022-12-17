#ifndef INCLUDESW_H_
#define INCLUDESW_H_
#include <iostream>
#include <stdlib.h>

#define dtype float

const int Tm_sw=20;
const int Tn_sw=16;
const int Tr_sw=20;
const int Tc_sw=20;
const int R_K_C_sw =1;
const int C_K_C_sw =150;


void compute_conv_sw(float input_fm_tile[Tn_sw][(Tr_sw-1)*1+R_K_C_sw][(Tc_sw-1)*1+C_K_C_sw], float weights_tile[Tm_sw][Tn_sw][R_K_C_sw][C_K_C_sw], float output_fm_tile [Tm_sw][Tr_sw][Tc_sw], float bias_tiled[Tm_sw], int row, int col, int to, int ti, int RO,int CO,int M, int N, int S, int RK,int CK);
void perform_conv_tiled_main_sw(float *input,float *output,int row,int col,int to,int ti, int N,int RO,int CO,float *weight, float *bias,int RK,int CK,int M,int S, int P,int RI,int CI,int row_in, int col_in);
void perform_conv_sw(float* input, float* output, float* weight, float* bias, int N, int M, int RI,int CI,int RO,int CO,int RK,int CK,int S, int P);



void perform_dense_sw (dtype* input, dtype* output, const dtype* weight, const dtype* bias, int M, int N, int Relu);

#endif
