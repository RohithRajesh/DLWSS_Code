#ifndef MODEL_H_
#define MODEL_H_

#define MAX_FMAP_CONV 14*299*256
#define OUTPUT_BUFFER_SIZE 14*299*256
#define INPUT_BUFFER_SIZE 14*299*256
#define INPUT_FILE_IMAGE 14*299*2
//#define FILTER_SIZE 5
#define SIZE_WCONV1 76800
#define SIZE_BCONV1 256
#define SIZE_WCONV2 3276800
#define SIZE_BCONV2 128
#define SIZE_WCONV3 417792
#define SIZE_BCONV3 64
#define SIZE_WF1 12544
#define SIZE_BF1 14
#define W_FILE_SIZE 3784398 // 150x256x2+256+100x128x256+128+51x64x128+64+64x14x14+14

/*
Conv Layer 		--> {0, #input Featuremaps,  #output Featuremaps,Input Row,Input Col,Output Row,Output Col,Filter Row,Filter Col}
FC Layer   		--> {2, Size of Input, Size of Output, RELU, 0 , 0 }
*/
unsigned layer[4][9] = {{0,2,256,14,299,14,150,1,150}, //Conv1
						{0,256,128,14,150,14,51,1,100}, //Conv2
						{0,128,64,14,51,14,1,1,51}, //Conv3
						{2,896,14,0,0,0}, //FC1
						};
#endif


// unsigned layer[7][6] = {{0,1,6,28,24,5},
// 						{1,6,6,24,12,2},
// 						{0,6,16,12,8,5},
// 						{},
// 						{},
// 						{},
// 						{},
// 						};
// #endif
