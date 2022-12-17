
#include <stdio.h>
#include "include_sw.h"
#include <iostream>
#include <stdlib.h>
 #ifdef __SDSOC

 	using namespace std;
 	#include "sds_lib.h"
 #endif
#define min(a,b) ((a) < (b) ? a : b)
//static int count=0;


void load_input_tile_sw(float *input, float input_fm_tile[Tn_sw][(Tr_sw-1)*1+R_K_C_sw][(Tc_sw-1)*1+C_K_C_sw], int row_in, int col_in, int ti, int S, int RK,int CK,int N, int RI,int CI){
 // #pragma HLS inline
//#pragma HLS PIPELINE
	const int max_1R= (Tr_sw-1)*1+1;
	const int max_1C=(Tc_sw-1)*1+150;
load_input:	for (int i=ti; i< min(ti+Tn_sw,N); i++){
 #pragma HLS loop_tripcount min=1 max=Tn_sw
	    for (int j=row_in; j<min(row_in+((Tr_sw-1)*S+RK),RI); j++){
	    	// for (int j=row; j<min(row+((Tr_sw-1)*S+K+1),I); j++){
 #pragma HLS loop_tripcount min=1 max=max_1R
	      for (int k=col_in; k<min(col_in+((Tc_sw-1)*S+CK),CI); k++){
	    		// for (int k=col; k<min(col+((Tc_sw-1)*S+K+1),I); k++){
 #pragma HLS loop_tripcount min=1 max=max_1C
 // #pragma HLS LOOP_MERGE
  #pragma HLS PIPELINE
//	       input_fm_tile[i][j][k] = input[I*I*(ti+i)+(row+j)*I+(col+k)];
	       input_fm_tile[i-ti][j-row_in][k-col_in] = input[RI*CI*i+j*CI+k];
//	       std::cout<<"Input: I*I*(ti+i)+(row+j)*I+(col+k) = "<<I*I*(ti+i)+(row+j)*I+(col+k)<<std::endl;
	       // std::cout<<"I*I*i+j*I+k = "<<I*I*i+j*I+k<<std::endl;
	      }
	    }
	  }
}

void load_weight_tile_sw(float *weight, float weights_tile[Tm_sw][Tn_sw][R_K_C_sw][C_K_C_sw], int to, int ti, int M, int N, int RK,int CK){
 // #pragma HLS inline
//	#pragma HLS PIPELINE
	load_weight:	for (int i=to;i<min(to+Tm_sw,M); i++){
	 #pragma HLS loop_tripcount min=1 max=Tm_sw
		int i_temp=i*N*RK*CK;
		for (int j=ti; j<min(ti+Tn_sw,N); j++){

	       #pragma HLS loop_tripcount min=1 max=Tn_sw
			int j_temp=j*RK*CK;
		   for (int k=0; k<RK; k++){
	         #pragma HLS loop_tripcount min=1 max=1
			   int k_temp=k*CK;
	    	 for (int l=0; l<CK; l++){
#pragma HLS loop_tripcount min=51 max=150
	          // #pragma HLS loop_tripcount min=3 max=11
// #pragma HLS LOOP_MERGE
#pragma HLS PIPELINE
	          weights_tile[i-to][j-ti][k][l] = weight[i_temp+j_temp+k_temp+l];
//	          std::cout<<"Weight: (to+i)*N*K*K+(ti+j)*K*K+(k*K)+l ="<<(to+i)*N*K*K+(ti+j)*K*K+(k*K)+l<<std::endl;
	        }
	      }
	   }
	}
}

void load_bias_tile_sw(float *bias, float bias_tiled[Tm_sw], int to, int M){
 // #pragma HLS inline
for (int i=to; i<min(to+Tm_sw,M); i++){
#pragma HLS loop_tripcount min=Tm_sw max =Tm_sw
#pragma HLS PIPELINE II=1
	if(i<min(to+Tm_sw,M)){

	 // #pragma HLS LOOP_MERGE

		 bias_tiled[i-to]= bias[i];
		}
	}
}


void compute_conv_sw(float input_fm_tile[Tn_sw][(Tr_sw-1)*1+R_K_C_sw][(Tc_sw-1)*1+C_K_C_sw], float weights_tile[Tm_sw][Tn_sw][R_K_C_sw][C_K_C_sw], float output_fm_tile [Tm_sw][Tr_sw][Tc_sw], float bias_tiled[Tm_sw], int row, int col, int to, int ti, int RO,int CO,int M, int N, int S, int RK,int CK){
 // #pragma HLS inline
	//		            float temp  = 0.0;
//#pragma HLS PIPELINE
//float partial_mul[Tm_sw][Tn_sw];

//  #pragma HLS ARRAY_PARTITION variable=partial_mul complete dim=1
//  #pragma HLS ARRAY_PARTITION variable=partial_mul complete dim=2



// uint Kernel_size_2b  = K;
	// unsigned char TR_MIN_uc = Tr_sw;
	// unsigned char TC_MIN_uc = Tc_sw;

for(int i =0;i < RK; i++){
 #pragma HLS loop_tripcount min=1 max=1
		for(int j = 0;j < CK; j++){
 #pragma HLS loop_tripcount min=50 max=150
			for(int trr = 0;trr < Tr_sw;trr++){
 #pragma HLS loop_tripcount min=1 max=Tr_sw

				for(int tcc = 0;tcc < Tc_sw;tcc++)
				{
#pragma HLS PIPELINE
 #pragma HLS loop_tripcount min=1 max=Tc_sw


					if((trr+row < RO)&&(tcc+col < CO)){
					for(int too = 0;too < Tm_sw;too++){
						if(i==0 &&j==0&&ti==0)output_fm_tile[too][trr][tcc]=bias_tiled[too];
						for(int tii = 0;tii <Tn_sw;tii++)
						{
//						#pragma HLS pipeline
							output_fm_tile[too][trr][tcc]+=weights_tile[too][tii][i][j]* input_fm_tile[tii][S*(trr)+i][S*(tcc)+j];
							// returns[1]+=partial_mul[too][tii];
						}
					}

					}
				}
			}}}
}



void store_output_sw(float *output, float output_fm_tile[Tm_sw][Tr_sw][Tc_sw], int row, int col,int to, int M, int RO,int CO){
// #pragma HLS inline
int col_hw=col;
int row_hw=row;

// pipelining gives wrong output somehow.
//	static int count=0;
	for (int i=0; to+i<min(to+Tm_sw,M); i++){
			 #pragma HLS loop_tripcount min=1 max=Tm_sw
		for (int j=0; row_hw+j<min(row_hw+Tr_sw,RO); j++){
			 #pragma HLS loop_tripcount min=1 max=Tr_sw
			for (int k=0; col_hw+k<min(col_hw+Tc_sw,CO); k++){
				 // #pragma HLS PIPELINE
       			 #pragma HLS loop_tripcount min=1 max=Tc_sw
       			// output[((to+i)*RO*CO)+((row+j)*CO)+(col+k)] =  (output_fm_tile[i][j][k]>0) ? output_fm_tile[i][j][k]:0;
       			int x=(to+i)*RO*CO;
       			int y=(row_hw+j)*CO;
       			int z=(col_hw+k);
				output[x+y+z] =  (output_fm_tile[i][j][k]>0) ? output_fm_tile[i][j][k]:0;

//       			returns[count]=x;
//       			returns[count+12]=j;
//       			returns[count+24]=k;
//       			count++;

       			// std::cout<<output[((to+i)*RO*CO)+((row+j)*CO)+(col+k)]<<std::endl;
				// output[((to+i)*min(row+Tr_sw,O)*min(col+Tc_sw,O))+((row+j)*min(col+Tc_sw,O))+(col+k)] =  output_fm_tile[i][j][k];
				// output[((to+i)*Tr_sw*Tc_sw)+((row+j)*Tc_sw)+(col+k)] =  output_fm_tile[i][j][k];
//				output_fm_tile[i][j][k]=float(0);
			}
		}
	}
//	returns[count]=-1;
//	count++;


}

void perform_conv_sw(float* input, float* output, float* weight, float* bias, int N, int M, int RI,int CI,int RO,int CO,int RK,int CK,int S, int P){
//#pragma HLS PIPELINE
int row_in, col_in;
	Top_Loop0 :for(int to=0; to<M; to+=Tm_sw){ // for all output channels with a step of Tm_sw
 #pragma HLS loop_tripcount min=14/Tm_sw max=14/Tm_sw
//			assert(to<=14/Tm_sw);

			row_in = 0;
			top_loop1: for(int row=0; row<RO; row+=Tr_sw){ //for all rows with a step of Tr_sw
 #pragma HLS loop_tripcount min=14/Tr_sw max=14/Tr_sw
//					assert(row<=14/Tr_sw);
					col_in = 0;
					top_loop2: for(int col=0;col<CO; col+=Tc_sw){ // for all columns with  a step Tc_sw
 #pragma HLS loop_tripcount min=51/Tc_sw max=299/Tc_sw
//						assert(col<=299/Tc_sw);
				top_loop3: for(int ti=0; ti<N; ti+=Tn_sw){
 #pragma HLS loop_tripcount min=64/Tn_sw max=256/Tn_sw
//						assert(ti<=256/Tn_sw);
// #pragma HLS pipeline
					// returns[1]+=1;
//					std::cout<<"Herre"<<std::endl;
					perform_conv_tiled_main_sw(input,output,row,col,to,ti, N, RO,CO,weight, bias, RK,CK,M,S,P,RI,CI, row_in, col_in);
				}
				// col_in += (Tc_sw-1)*S+K;
				col_in += Tc_sw*S;
				// if(col==2) {exit(0);};
			}
			// row_in += (Tr_sw-1)*S+K;
			row_in += Tr_sw*S;
		}
	}


}

void perform_conv_tiled_main_sw(float *input,float *output,int row,int col,int to,int ti, int N,int RO,int CO,float *weight, float *bias,int RK,int CK,int M,int S, int P,int RI,int CI,int row_in, int col_in){
	//-------initialize all on-chip buffers----------------------------------
//	#pragma HLS PIPELINE
//	 #pragma HLS DATAFLOW
//#pragma HLS inline region
	float output_fm_tile [Tm_sw][Tr_sw][Tc_sw];

	// float input_fm_tile [Tn_sw][(Tr_sw-1)*4+R_K_C_sw][(Tc_sw-1)*4+C_K_C_sw];//={{{0}}}; //input tile

	// float weights_tile[Tm_sw][Tn_sw][R_K_C_sw][C_K_C_sw];//=; //weights tile

	// float bias_tiled[Tm_sw];//={0}
	static float input_fm_tile [Tn_sw][(Tr_sw-1)*1+R_K_C_sw][(Tc_sw-1)*1+C_K_C_sw]; //input tile

	static float weights_tile[Tm_sw][Tn_sw][R_K_C_sw][C_K_C_sw]; //weights tile

	static float bias_tiled[Tm_sw];

//
 	#pragma HLS ARRAY_PARTITION variable=output_fm_tile dim=1 complete
 	#pragma HLS ARRAY_PARTITION variable=input_fm_tile dim=1 complete
 	#pragma HLS ARRAY_PARTITION variable=weights_tile dim=1 complete
 	#pragma HLS ARRAY_PARTITION variable=weights_tile dim=2 complete
 	#pragma HLS ARRAY_PARTITION variable=bias_tiled dim=1 complete


		for(int i=0;i<Tm_sw;i++){
			for(int j=0;j<Tn_sw;j++){
				for(int k=0;k<R_K_C_sw;k++){
					for(int l=0;l<C_K_C_sw;l++){
#pragma HLS pipeline II=1

						weights_tile[i][j][k][l]=0;
					}
				}
			}
		}
		for(int i=0;i<Tm_sw;i++){
#pragma HLS pipeline II=1

			bias_tiled[i]=0;
		}
		for(int i=0;i<Tn_sw;i++){
			for(int j=0;j<(Tr_sw-1)*1+R_K_C_sw;j++){
				for(int k=0;k<(Tc_sw-1)*1+C_K_C_sw;k++){
#pragma HLS pipeline II=1
					input_fm_tile[i][j][k]=0;
				}
			}
		}

// //#pragma HLS PIPELINE

//		#pragma HLS DATAFLOW



	//--------load input chunk---------------------------------------------------
	load_input_tile_sw(input,input_fm_tile,row_in,col_in,ti,S,RK,CK,N,RI,CI);





	//---------load weights chunk------------------------------------------------
	load_weight_tile_sw(weight,weights_tile,to,ti,M,N,RK,CK);


	//------ load bias values--------------------------------------------------
	load_bias_tile_sw(bias,bias_tiled,to,M);

	//----------------Compute the convolution for the tile-----------------------------------------------
	compute_conv_sw(input_fm_tile,weights_tile,output_fm_tile,bias_tiled,row,col,to,ti,RO,CO,M,N,S,RK,CK);



	//----------- if all input channels are processed for a tile-----------------------------------------
	if ((ti+Tn_sw)>=N-1){
	//------------- add bias and perform ReLU -----------------------------------------------------------
//		add_bias_relu(output_fm_tile,bias_tiled,row,col,to,O,M);
	//----------------- store the output tile to DDR and initialize output_fm_tile-----------------------
		store_output_sw(output, output_fm_tile, row, col,to, M, RO,CO);
	}



}

