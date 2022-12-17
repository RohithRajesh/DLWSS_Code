#include "include_sw.h"

#include <math.h>
// M --> # of input channles/Neurons
// N --> # of output channels/neurons
// O --> Output Image Size
// I --> Input Image Size
// K --> Filter Size

// FC layer y = WX + b
void perform_dense_sw (dtype* input, dtype* output, const dtype* weight, const dtype* bias, int M, int N, int Relu)
{





 for (int n = 0; n < N; n++) { // for all outputs
  dtype temp = 0.0f;
  for (int m = 0; m < M; m++) { // for all inputs
    int w_index = m + n * M; // Since weight is a 1-D Pointer, calculating the pointer index.
    temp += input[m] * weight[w_index]; //WX
  }
  temp = temp + bias[n]; // + bias
  if (Relu == 1)
    temp = (temp > float(0)) ? temp : float(0); // ReLU
  else if(Relu==0)
    temp= 1/ float(1+ exp(-temp)); // Sigmoid
  output[n] = temp;


}
}



