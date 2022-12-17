#include <stdio.h>
#include "include.h"
#include <iostream>
#include <stdlib.h>
 #ifdef __SDSOC

 	using namespace std;
 	#include "sds_lib.h"
 #endif
#define min(a,b) ((a) < (b) ? a : b)
//static int count=0;


void load_input_tile(float *input, ftype input_fm_tile[Tn][(Tr-1)*1+R_K_C][(Tc-1)*1+C_K_C], int row_in, int col_in, int ti, int S, int RK,int CK,int N, int RI,int CI){
  #pragma HLS inline
//#pragma HLS PIPELINE
	const int max_1R= (Tr-1)*1+1;
	const int max_1C=(Tc-1)*1+150;
load_input:	for (int i=ti; i< min(ti+Tn,N); i++){
 #pragma HLS loop_tripcount min=1 max=Tn
	    for (int j=row_in; j<min(row_in+((Tr-1)*S+RK),RI); j++){
	    	// for (int j=row; j<min(row+((Tr-1)*S+K+1),I); j++){
 #pragma HLS loop_tripcount min=1 max=max_1R
	      for (int k=col_in; k<min(col_in+((Tc-1)*S+CK),CI); k++){
	    		// for (int k=col; k<min(col+((Tc-1)*S+K+1),I); k++){
 #pragma HLS loop_tripcount min=1 max=max_1C
 // #pragma HLS LOOP_MERGE
  #pragma HLS PIPELINE
//	       input_fm_tile[i][j][k] = input[I*I*(ti+i)+(row+j)*I+(col+k)];
	       input_fm_tile[i-ti][j-row_in][k-col_in] = (ftype)input[RI*CI*i+j*CI+k];
//	       std::cout<<"Input: I*I*(ti+i)+(row+j)*I+(col+k) = "<<I*I*(ti+i)+(row+j)*I+(col+k)<<std::endl;
	       // std::cout<<"I*I*i+j*I+k = "<<I*I*i+j*I+k<<std::endl;
	      }
	    }
	  }
}

void load_weight_tile(float *weight, ftype_w weights_tile[Tm][Tn][R_K_C][C_K_C], int to, int ti, int M, int N, int RK,int CK){
  #pragma HLS inline
//	#pragma HLS PIPELINE
	load_weight:	for (int i=to;i<min(to+Tm,M); i++){
	 #pragma HLS loop_tripcount min=1 max=Tm
		int i_temp=i*N*RK*CK;
		for (int j=ti; j<min(ti+Tn,N); j++){

	       #pragma HLS loop_tripcount min=1 max=Tn
			int j_temp=j*RK*CK;
		   for (int k=0; k<RK; k++){
#pragma HLS PIPELINE
	         #pragma HLS loop_tripcount min=1 max=1
			   int k_temp=k*CK;
	    	 for (int l=0; l<CK; l++){
#pragma HLS loop_tripcount min=51 max=150
	          // #pragma HLS loop_tripcount min=3 max=11
// #pragma HLS LOOP_MERGE
//#pragma HLS PIPELINE
	          weights_tile[i-to][j-ti][k][l] = (ftype_w)weight[i_temp+j_temp+k_temp+l];
//	          std::cout<<"Weight: (to+i)*N*K*K+(ti+j)*K*K+(k*K)+l ="<<(to+i)*N*K*K+(ti+j)*K*K+(k*K)+l<<std::endl;
	        }
	      }
	   }
	}
}

void load_bias_tile(float *bias, ftype_w bias_tiled[Tm], int to, int M){
  #pragma HLS inline
for (int i=to; i<min(to+Tm,M); i++){
#pragma HLS loop_tripcount min=Tm max =Tm
#pragma HLS PIPELINE II=1
	if(i<min(to+Tm,M)){

	 // #pragma HLS LOOP_MERGE

		 bias_tiled[i-to]= (ftype)bias[i];
		}
	}
}


void compute_conv(ftype input_fm_tile[Tn][(Tr-1)*1+R_K_C][(Tc-1)*1+C_K_C], ftype_w weights_tile[Tm][Tn][R_K_C][C_K_C], ftype output_fm_tile [Tm][Tr][Tc], ftype_w bias_tiled[Tm], int row, int col, int to, int ti, int RO,int CO,int M, int N, int S, int RK,int CK){
  #pragma HLS inline
	//		            float temp  = 0.0;
//#pragma HLS PIPELINE
//float partial_mul[Tm][Tn];

//  #pragma HLS ARRAY_PARTITION variable=partial_mul complete dim=1
//  #pragma HLS ARRAY_PARTITION variable=partial_mul complete dim=2



// uint Kernel_size_2b  = K;
	// unsigned char TR_MIN_uc = Tr;
	// unsigned char TC_MIN_uc = Tc;

for(int i =0;i < RK; i++){
 #pragma HLS loop_tripcount min=1 max=1
		for(int j = 0;j < CK; j++){
 #pragma HLS loop_tripcount min=50 max=150
//#pragma HLS pipeline
			for(int trr = 0;trr < Tr;trr++){
 #pragma HLS loop_tripcount min=1 max=Tr

				for(int tcc = 0;tcc < Tc;tcc++)
				{
#pragma HLS PIPELINE
 #pragma HLS loop_tripcount min=1 max=Tc


					if((trr+row < RO)&&(tcc+col < CO)){
					for(int too = 0;too < Tm;too++){
						if(i==0 &&j==0&&ti==0)output_fm_tile[too][trr][tcc]=bias_tiled[too];
						for(int tii = 0;tii <Tn;tii++)
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



void store_output(float *output, ftype output_fm_tile[Tm][Tr][Tc], int row, int col,int to, int M, int RO,int CO){
 #pragma HLS inline
int col_hw=col;
int row_hw=row;

// pipelining gives wrong output somehow.
//	static int count=0;
	for (int i=0; to+i<min(to+Tm,M); i++){
			 #pragma HLS loop_tripcount min=1 max=Tm
		for (int j=0; row_hw+j<min(row_hw+Tr,RO); j++){
			 #pragma HLS loop_tripcount min=1 max=Tr
			for (int k=0; col_hw+k<min(col_hw+Tc,CO); k++){
				 // #pragma HLS PIPELINE
       			 #pragma HLS loop_tripcount min=1 max=Tc
       			// output[((to+i)*RO*CO)+((row+j)*CO)+(col+k)] =  (output_fm_tile[i][j][k]>0) ? output_fm_tile[i][j][k]:0;
       			int x=(to+i)*RO*CO;
       			int y=(row_hw+j)*CO;
       			int z=(col_hw+k);
				output[x+y+z] =  (output_fm_tile[i][j][k]>(ftype)0) ? (float)output_fm_tile[i][j][k]:(float)0;

//       			returns[count]=x;
//       			returns[count+12]=j;
//       			returns[count+24]=k;
//       			count++;

       			// std::cout<<output[((to+i)*RO*CO)+((row+j)*CO)+(col+k)]<<std::endl;
				// output[((to+i)*min(row+Tr,O)*min(col+Tc,O))+((row+j)*min(col+Tc,O))+(col+k)] =  output_fm_tile[i][j][k];
				// output[((to+i)*Tr*Tc)+((row+j)*Tc)+(col+k)] =  output_fm_tile[i][j][k];
//				output_fm_tile[i][j][k]=float(0);
			}
		}
	}
//	returns[count]=-1;
//	count++;


}

void perform_conv(float* input, float* output, float* weight, float* bias, int N, int M, int RI,int CI,int RO,int CO,int RK,int CK,int S, int P){
//#pragma HLS PIPELINE
int row_in, col_in;
	Top_Loop0 :for(int to=0; to<M; to+=Tm){ // for all output channels with a step of Tm
 #pragma HLS loop_tripcount min=14/Tm max=14/Tm
//			assert(to<=14/Tm);

			row_in = 0;
			top_loop1: for(int row=0; row<RO; row+=Tr){ //for all rows with a step of Tr
 #pragma HLS loop_tripcount min=14/Tr max=14/Tr
//					assert(row<=14/Tr);
					col_in = 0;
					top_loop2: for(int col=0;col<CO; col+=Tc){ // for all columns with  a step Tc
 #pragma HLS loop_tripcount min=51/Tc max=299/Tc
//						assert(col<=299/Tc);
				top_loop3: for(int ti=0; ti<N; ti+=Tn){
 #pragma HLS loop_tripcount min=64/Tn max=256/Tn
//						assert(ti<=256/Tn);
// #pragma HLS pipeline
					// returns[1]+=1;
//					std::cout<<"Herre"<<std::endl;
					perform_conv_tiled_main(input,output,row,col,to,ti, N, RO,CO,weight, bias, RK,CK,M,S,P,RI,CI, row_in, col_in);
				}
				// col_in += (Tc-1)*S+K;
				col_in += Tc*S;
				// if(col==2) {exit(0);};
			}
			// row_in += (Tr-1)*S+K;
			row_in += Tr*S;
		}
	}


}

void perform_conv_tiled_main(float *input,float *output,int row,int col,int to,int ti, int N,int RO,int CO,float *weight, float *bias,int RK,int CK,int M,int S, int P,int RI,int CI,int row_in, int col_in){
	//-------initialize all on-chip buffers----------------------------------
//	#pragma HLS PIPELINE
//	 #pragma HLS DATAFLOW
#pragma HLS inline
	ftype output_fm_tile [Tm][Tr][Tc];

	// float input_fm_tile [Tn][(Tr-1)*4+R_K_C][(Tc-1)*4+C_K_C];//={{{0}}}; //input tile

	// float weights_tile[Tm][Tn][R_K_C][C_K_C];//=; //weights tile

	// float bias_tiled[Tm];//={0}
	static ftype input_fm_tile [Tn][(Tr-1)*1+R_K_C][(Tc-1)*1+C_K_C]; //input tile

	static ftype_w weights_tile[Tm][Tn][R_K_C][C_K_C]; //weights tile

	static ftype_w bias_tiled[Tm];

//
 	#pragma HLS ARRAY_PARTITION variable=output_fm_tile dim=1 complete
 	#pragma HLS ARRAY_PARTITION variable=input_fm_tile dim=1 complete
 	#pragma HLS ARRAY_PARTITION variable=weights_tile dim=1 complete
 	#pragma HLS ARRAY_PARTITION variable=weights_tile dim=2 complete
 	#pragma HLS ARRAY_PARTITION variable=bias_tiled dim=1 complete


		for(int i=0;i<Tm;i++){
			for(int j=0;j<Tn;j++){
				for(int k=0;k<R_K_C;k++){
					for(int l=0;l<C_K_C;l++){
#pragma HLS pipeline II=1

						weights_tile[i][j][k][l]=0;
					}
				}
			}
		}
		for(int i=0;i<Tm;i++){
#pragma HLS pipeline II=1
			bias_tiled[i]=0;
		}
		for(int i=0;i<Tn;i++){
			for(int j=0;j<(Tr-1)*1+R_K_C;j++){
				for(int k=0;k<(Tc-1)*1+C_K_C;k++){
#pragma HLS pipeline II=1
					input_fm_tile[i][j][k]=0;
				}
			}
		}

// //#pragma HLS PIPELINE

//		#pragma HLS DATAFLOW



	//--------load input chunk---------------------------------------------------
	load_input_tile(input,input_fm_tile,row_in,col_in,ti,S,RK,CK,N,RI,CI);





	//---------load weights chunk------------------------------------------------
	load_weight_tile(weight,weights_tile,to,ti,M,N,RK,CK);


	//------ load bias values--------------------------------------------------
	load_bias_tile(bias,bias_tiled,to,M);

	//----------------Compute the convolution for the tile-----------------------------------------------
	compute_conv(input_fm_tile,weights_tile,output_fm_tile,bias_tiled,row,col,to,ti,RO,CO,M,N,S,RK,CK);



	//----------- if all input channels are processed for a tile-----------------------------------------
	if ((ti+Tn)>=N-1){
	//------------- add bias and perform ReLU -----------------------------------------------------------
//		add_bias_relu(output_fm_tile,bias_tiled,row,col,to,O,M);
	//----------------- store the output tile to DDR and initialize output_fm_tile-----------------------
		store_output(output, output_fm_tile, row, col,to, M, RO,CO);
	}



}

