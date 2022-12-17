#include "include.h"

#include <math.h>
// M --> # of input channles/Neurons
// N --> # of output channels/neurons
// O --> Output Image Size
// I --> Input Image Size
// K --> Filter Size

// FC layer y = WX + b

void perform_dense (dtype *input, dtype *output, const dtype *weight, const dtype *bias,int Relu)
{
	ftype localIn[896];
	ftype localOut[14];
	ftype_w localweight[14*896];

	for(int i=0;i<896;i++){
#pragma HLS pipeline
		localIn[i]=(ftype)input[i];
	}
	for(int i=0;i<14;i++){
#pragma HLS pipeline
		localOut[i]=(ftype)output[i];
	}
	for(int i=0;i<14*896;i++){
#pragma HLS pipeline
		localweight[i]=(ftype_w)weight[i];
	}
#pragma HLS array_partition variable=localIn complete
//#pragma HLS array_partition variable=localweight complete
#pragma HLS array_partition partition variable=localOut complete
//#pragma HLS array_partition variable=output complete




 for (int n = 0; n < 14; n++) { // for all outputs

  ftype temp=0.0f;
  for (int m = 0; m < 896; m++) { // for all inputs
#pragma HLS pipeline
    temp += localIn[m] * localweight[m + n *896]; //WX
  }
//  int temp=output[n]
  temp=temp+(ftype)bias[n];
//  if (Relu == 1)
//   temp = (ftype)((temp > ftype(0)) ? temp : ftype(0)); // ReLU
//  else if(Relu==0)
//    temp= (ftype)(1/ dtype(1+ exp(-(float)temp))); // Sigmoid
  localOut[n]=temp;

}

 for(int i=0;i<14;i++){
#pragma HLS pipeline
	 output[i]=float(localOut[i]);
 }
}
//void perform_dense_sw (dtype* input, dtype* output, const dtype* weight, const dtype* bias, int M, int N, int Relu)
//{
//
//
//
//
//
// for (int n = 0; n < N; n++) { // for all outputs
//  dtype temp = 0.0f;
//  for (int m = 0; m < M; m++) { // for all inputs
//    int w_index = m + n * M; // Since weight is a 1-D Pointer, calculating the pointer index.
//    temp += input[m] * weight[w_index]; //WX
//  }
//  temp = temp + bias[n]; // + bias
//  if (Relu == 1)
//    temp = (temp > float(0)) ? temp : float(0); // ReLU
//  else if(Relu==0)
//    temp= 1/ float(1+ exp(-temp)); // Sigmoid
//  output[n] = temp;
//
//
//}
//}
