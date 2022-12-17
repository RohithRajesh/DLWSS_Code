#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <fstream>
#include "model.h"
#include "sds_lib.h"
#include "include.h"
#include "include_sw.h"
#include<time.h>
using namespace std;
#define TESTCASES 1400
#define ENABLE_TIMER 0
//#define sds_alloc malloc
//#define sds_free free
//#define sds_free free

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> v;
  unsigned int i = 0;
  unsigned int j = s.find(c);
  while (j < s.size() ){
    v.push_back(s.substr(i, j - i));
    i = ++j;
    j = s.find(c, j);
    if (j >= s.size() ) {
      v.push_back(s.substr(i,s.size() ));
      break;
    }
  }
  return v;
}

int difference(int* predictions,int* predictions_sw){
  float acc=0;
  for(int i=0;i<14;i++)acc+=std::abs(predictions[i]-predictions_sw[i]);
  return acc;

}

void sigmoid(float* output){
	for(int i=0;i<14;i++){
		output[i]=1/ float(1+ exp(-output[i]));
	}
}



int main(int argc, char** argv)
{

  struct timespec start, end;
  double seconds,nanoseconds,elapsed,elapsed_hw;
 
  int label[14];

  std::cout<<"Initializing Model"<<std::endl;



 	dtype *input_buffer      = (dtype*) sds_alloc(sizeof(dtype)*INPUT_BUFFER_SIZE);
  dtype *output_buffer     = (dtype*) sds_alloc(sizeof(dtype)*OUTPUT_BUFFER_SIZE);
  dtype *input_file_buffer = (dtype*) sds_alloc(sizeof(dtype)*INPUT_FILE_IMAGE);
#if ENABLE_TIMER
		dtype *input_buffer_sw      = (dtype*) sds_alloc(sizeof(dtype)*INPUT_BUFFER_SIZE);
  dtype *output_buffer_sw     = (dtype*) sds_alloc(sizeof(dtype)*OUTPUT_BUFFER_SIZE);
  dtype *input_file_buffer_sw = (dtype*) sds_alloc(sizeof(dtype)*INPUT_FILE_IMAGE);
#endif
  dtype *weight_buffer     = (dtype*) sds_alloc(sizeof(dtype)*W_FILE_SIZE);
  dtype *w_conv1           = (dtype*) sds_alloc(sizeof(dtype)*SIZE_WCONV1);
  dtype *b_conv1           = (dtype*) sds_alloc(sizeof(dtype)*SIZE_BCONV1);
  dtype *w_conv2           = (dtype*) sds_alloc(sizeof(dtype)*SIZE_WCONV2 );
  dtype *b_conv2           = (dtype*) sds_alloc(sizeof(dtype)*SIZE_BCONV2 );
  dtype *w_conv3             = (dtype*)sds_alloc(sizeof(dtype)*SIZE_WCONV3 );
  dtype *b_conv3             = (dtype*)sds_alloc(sizeof(dtype)*SIZE_BCONV3 );
  dtype *w_fc1             = (dtype*) sds_alloc(sizeof(dtype)*SIZE_WF1 );
  dtype *b_fc1             = (dtype*) sds_alloc(sizeof(dtype)*SIZE_BF1 );

	for(int i=0;i<INPUT_BUFFER_SIZE;i++){
		input_buffer[i]=0;
#if ENABLE_TIMER
	input_buffer_sw[i]=0;
#endif
	}
	for(int i=0;i<OUTPUT_BUFFER_SIZE;i++){
		output_buffer[i]=0;
#if ENABLE_TIMER
		output_buffer_sw[i]=0;
#endif
	}




    // Check for failed memory allocation
    if((input_buffer == NULL) || (output_buffer == NULL) || (input_file_buffer == NULL)){
      std::cout << "TEST FAILED : Failed to allocate memory" << std::endl;
      return -1;
    }

    // loading weights into DDR Memory
    std::ifstream in("weightsDat_final.txt",std::ios_base::binary);

    in.read((char *)weight_buffer,sizeof(float)*W_FILE_SIZE);

    int full=0;
    for(int i =0 ; i<SIZE_WCONV1; i++) // 5*5*1*6

      w_conv1[i] = dtype(weight_buffer[i]);


    full+=SIZE_WCONV1;


    for(int i =0 ; i<SIZE_BCONV1;i++)
      b_conv1[i] = dtype(weight_buffer[full+i]);

    full+=SIZE_BCONV1;

    for(int i =0 ; i<SIZE_WCONV2;i++) //5*5*6*16
      w_conv2[i] = dtype(weight_buffer[full+i]);

    full+=SIZE_WCONV2;
    for(int i =0 ; i<SIZE_BCONV2;i++)
      b_conv2[i] = dtype(weight_buffer[full+i]);

    full+=SIZE_BCONV2;

    for(int i =0;i<SIZE_WCONV3;i++) //256*120
      w_conv3[i] = dtype(weight_buffer[full+i]);
    full+=SIZE_WCONV3;
    for(int i =0 ; i<SIZE_BCONV3;i++)
      b_conv3[i] = dtype(weight_buffer[full+i]);

    full+=SIZE_BCONV3;
    for(int i =0 ; i<SIZE_WF1;i++) //120*84
      w_fc1[i] = dtype(weight_buffer[full+i]);

    full+=SIZE_WF1;
    for(long i =0 ; i<SIZE_BF1;i++)
      b_fc1[i] = dtype(weight_buffer[full+i]);


    std::fstream inputi;

    std::fstream input_label;

    inputi.open("Dataset.dat");

     input_label.open("Dataset_label.dat");
    // std::cout<<"h"<<std::endl;
    std::string lincA;
    std::string lincB;

//    int testcases = 2; //3500
    int pred_1=0;
    int true_1=0;
    int pred_1_only1=0;
    int true_1_only1=0;

    for (int t = 0; t<TESTCASES;t++)
    {
      getline(inputi, lincA);
      getline(input_label, lincB);
      std::vector<std::string> listfilemax = split(lincA,',');
      std::vector<std::string> listfilemaxB = split(lincB,',');
      for (int l=0; l<INPUT_FILE_IMAGE; l++){
        input_buffer[l]= (dtype)atof(listfilemax.at(l).c_str());
//        if(l<=20)std::cout<<input_buffer[l]<<std::endl;
		#if ENABLE_TIMER
				input_buffer_sw[l]=input_buffer[l];
		#endif
        //         if(t==0 && l<20)cout<<input_buffer[l]<<endl;
      }

      for(int l=0;l<14;l++){
          label[l]=  atof(listfilemaxB.at(l).c_str());
//          std::cout<<label[l]<<std::endl;
      }


/// ------------------------------SW Implementation ----------------------------------------------------

#if ENABLE_TIMER
 clock_gettime(CLOCK_REALTIME, &start);

      perform_conv_sw(input_buffer_sw,output_buffer_sw,w_conv1,b_conv1,layer[0][1],layer[0][2],layer[0][3],layer[0][4],layer[0][5],layer[0][6],layer[0][7],layer[0][8],1,0);
       // for(int i=0;i<20;i++){
       //          cout<<output_buffer_sw[i]<<endl;
       //        }
       std::cout<<"conv1 done"<<std::endl;
      perform_conv_sw(output_buffer_sw,input_buffer_sw,w_conv2,b_conv2,layer[1][1],layer[1][2],layer[1][3],layer[1][4],layer[1][5],layer[1][6],layer[1][7],layer[1][8],1,0);
     perform_conv_sw(input_buffer_sw,output_buffer_sw,w_conv3,b_conv3,layer[2][1],layer[2][2],layer[2][3],layer[2][4],layer[2][5],layer[2][6],layer[2][7],layer[2][8],1,0);
     perform_dense_sw(output_buffer_sw,input_buffer_sw,w_fc1,b_fc1,layer[3][1],layer[3][2],layer[3][3]);

  clock_gettime(CLOCK_REALTIME, &end);
  seconds = end.tv_sec - start.tv_sec;
  nanoseconds = end.tv_nsec - start.tv_nsec;
  elapsed = (seconds + nanoseconds*1e-9)*1000;


	std::cout<<"Hardware Execution"<<std::endl;
  clock_gettime(CLOCK_REALTIME, &start);
#endif



//cout<<"Inout"<<endl;
//for(int i=0;i<20;i++){
//	cout<<input_buffer[i]<<endl;
//}
perform_conv(input_buffer,output_buffer,w_conv1,b_conv1,layer[0][1],layer[0][2],layer[0][3],layer[0][4],layer[0][5],layer[0][6],layer[0][7],layer[0][8],1,0);
//cout<<"Conv 1 out"<<endl;
//for(int i=0;i<20;i++){
//          cout<<output_buffer[i]<<endl;
//        }
perform_conv(output_buffer,input_buffer,w_conv2,b_conv2,layer[1][1],layer[1][2],layer[1][3],layer[1][4],layer[1][5],layer[1][6],layer[1][7],layer[1][8],1,0);
perform_conv(input_buffer,output_buffer,w_conv3,b_conv3,layer[2][1],layer[2][2],layer[2][3],layer[2][4],layer[2][5],layer[2][6],layer[2][7],layer[2][8],1,0);
perform_dense(output_buffer,input_buffer,w_fc1,b_fc1,0);
sigmoid(input_buffer);

#if ENABLE_TIMER
	clock_gettime(CLOCK_REALTIME, &end);
	seconds = end.tv_sec - start.tv_sec;
	nanoseconds = end.tv_nsec - start.tv_nsec;
	elapsed_hw = (seconds + nanoseconds*1e-9)*1000;
#endif

	 dtype max = input_buffer[0];
	 int outputs = 0;
	 for (int j = 0; j < 14; j++)
	 {
		 if (input_buffer[j] >=0.5)
		 {
			 outputs = j;
			 max = input_buffer[j];
		 }
//			std::cout<<input_buffer[j]<<std::endl;
	 }

	 std::cout<< t <<": Predicted "<<outputs<<" "<<std::endl;

		int predictions[14];
		int predictions_sw[14];

		for(int i=0;i<14;i++){
		  if(input_buffer[i]>=0.5)predictions[i]=1;
		  else predictions[i]=0;
#if ENABLE_TIMER
		  if(input_buffer_sw[i]>=0.5)predictions_sw[i]=1;
		  else predictions_sw[i]=0;
#endif
		}
		std::cout<<"Predictions"<<std::endl;
		for(int i=0;i<14;i++)std::cout<<predictions[i]<<" ";
		std::cout<<""<<std::endl;

		std::cout<<"Labels"<<std::endl;
		for(int i=0;i<14;i++){
			if(predictions[i]==1 && label[i]==1){
				pred_1_only1++;
				true_1_only1++;
			}
			else if(predictions[i]==0 && label[i]==1){
				true_1_only1++;
			}
			std::cout<<label[i]<<" ";
		}
		std:cout<<""<<std::endl;
		for (int i=0;i<14;i++){
			if(predictions[i]==label[i])pred_1++;
			true_1++;
		}
		std::cout<<""<<std::endl;
	    std::cout<<"Accuracy by exact matching= "<<pred_1/((float)true_1)<<std::endl;
	    std::cout<<"Accuracy by  only ones= "<< pred_1_only1/(float)true_1_only1<<std::endl;


#if ENABLE_TIMER
	 std::cout<<"SW Time= "<<elapsed<<"\n HW Time= "<<elapsed_hw<<std::endl;
	 std::cout<<"Difference between SW and HW= "<<difference(predictions,predictions_sw)<<std::endl;
	 std::cout<<"Speedup= "<<elapsed/elapsed_hw<<std::endl;
#endif


}




    sds_free(input_buffer);
    sds_free(output_buffer);
    sds_free(weight_buffer);
    sds_free(w_conv1);
    sds_free(b_conv1);
    sds_free(w_conv2);
     sds_free(b_conv2);
     sds_free(w_conv3);
      sds_free(b_conv3);
    sds_free(w_fc1);
    sds_free(b_fc1);
    sds_free(input_file_buffer);
#if ENABLE_TIMER
    sds_free(input_buffer_sw);
     sds_free(input_file_buffer_sw);
#endif







}


















//
//
//
//
//#include "conv_hw2.h"
//#include <iostream>
//#include <cstring>
//#include <stdlib.h>
//#include <bits/stdc++.h>
//#include <fstream>
//#include "functions_def.h"
//#include "sds_lib.h"
//using namespace std;
//
//
//
//int main(int argc, char** argv)
//{
//
//
//	float* input=(dtype*) sds_alloc(sizeof(dtype)*18);
//	float* w=(dtype*) sds_alloc(sizeof(dtype)*4);
//	float* b=(dtype*) sds_alloc(sizeof(dtype)*2);
//	input[0]=1;
//	input[1]=2;
//	input[2]=3;
//	input[3]=1;
//	input[4]=2;
//	input[5]=3;
//	input[6]=1;
//	input[7]=2;
//	input[8]=3;
//	input[9]=1;
//	input[10]=2;
//	input[11]=3;
//	input[12]=1;
//	input[13]=2;
//	input[14]=3;
//	input[15]=1;
//	input[16]=2;
//	input[17]=3;
//
//
//	w[0]=1;
//	w[1]=2;
//	w[2]=1;
//	w[3]=2;
//	b[0]=1;
//	b[1]=1;
//
////	float* output=(float*) malloc(sizeof(float)*16);
//
//	 dtype *output = (dtype*) sds_alloc(sizeof(dtype)*12);
//	float* returns = (float*) sds_alloc(sizeof(float)*100);
//
//
//
//
//
//
//
///// ------------------------------Layer CONV1----------------------------------------------------
//perform_conv(input,output,w,b,1,2,3,3,3,2,1,2,1,0,returns);
//
////for(int i=0;i<100;i++){
////	cout<<returns[i]<<endl;
////}
//cout<<"Printing Output"<<endl;
//
//
//
//for(int i =0;i<12;i++){
//	cout<<output[i]<<endl;
//}
//cout<<"Printing Debug"<<endl;
//for(int i =0;i<13;i++){
//	cout<<returns[i]<<endl;
//}
////cout<<returns[0]<<endl;
//
//}
