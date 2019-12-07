///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// version 3.0
// title: 基于cuda的神经网络手写数字识别的优化
// 算法：SGD
// date: 2018-12-01
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <windows.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define CHECK(call){																				\
						cudaError_t error = call;													\
						if (error != cudaSuccess)													\
						{																			\
							printf("error: file %s,line %d;", __FILE__, __LINE__);					\
							printf("coding:%d,reason:%s\n", error, cudaGetErrorString(error));		\
							exit(1);																\
						}																			\
					}
#define INPUTNODES 784   //输入层节点数
#define HIDDENNODES 200  //隐藏层节点数
#define OUTPUTNODES 10   //输出层节点数
#define TRAIN_SIZE 60000 //训练集样本数
#define TEST_SIZE 10000  //测试集样本数
#define LR 0.1           //学习率
//获取时间,单位秒
inline double seconds()
{
	LARGE_INTEGER nFreq;//LARGE_INTEGER在64位系统中是LONGLONG，在32位系统中是高低两个32位的LONG，在windows.h中通过预编译宏作定义
	LARGE_INTEGER nTime;//记录开始时的计数器的值
	double time;

	QueryPerformanceFrequency(&nFreq);//获取系统时钟频率
	QueryPerformanceCounter(&nTime);//获取开始时刻计数值
	time = (double)(nTime.QuadPart) / (double)nFreq.QuadPart;
	return time;
}
//struct NeuralNetwork {
//	float *Input; //[INPUTNODES]//输入层输入
//	float *Hidden;//[HIDDENNODES]//隐藏层输出
//	float *Output;//[OUTPUTNODES]//输出层输出
//	float *Target;//[OUTPUTNODES]//目标输出
//	float *Wih;//[HIDDENNODES][INPUTNODES]//输入层到隐藏层权值
//	float *Who;//[OUTPUTNODES][HIDDENNODES]//隐藏层到输出层权值
//};

//随机初始化权值在-0.5到0.5之间
void InitWeigth(float *pw1, float *pw2)
{
	time_t t;
	srand((unsigned int)time(&t));
	for (int i = 0; i < HIDDENNODES*INPUTNODES; i++)
	{
		pw1[i] = rand() / (float)(RAND_MAX)-0.5;
	}
	for (int i = 0; i < OUTPUTNODES*HIDDENNODES; i++)
	{
		pw2[i]= rand() / (float)(RAND_MAX)-0.5;
	}
}

//////数据处理//////////////
//目标向量化 one-hot
// label=1对于,list向量为[0,1 0.99 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
void label2list(int label, float *list, int N)
{
	for (int i = 0; i < N; i++)
	{
		if (i == label)
			list[i] = 0.99;
		else
			list[i] = 0.1;
	}
}

//one-hot转化为数字label
int list2label(float *list, int N)
{
	float max = 0.0;
	int label = 0;
	for (int i = 0; i < N; i++)
	{
		if (max <list[i])
		{
			max = list[i];
			label = i;
		}
	}
	return label;
}

//获取图片数据到inputImage中，图片目标向量到target_label
void getData(FILE *fp, float *inputImage, float *target_label)
{
	int label = 0;
	char *line, *record;
	char buffer[2048];
	if ((line = fgets(buffer, sizeof(buffer), fp)) != NULL)//当没有读取到文件末尾时继续
	{
		record = strtok(line, ",");
		int index = 0;
		while (record != NULL)//读取每一行的数据
		{
			if (index == 0)
			{
				label = atoi(record);
			}
			else
			{
				inputImage[index - 1] = ((float)(atoi(record)) / 255.0) * 0.99 + 0.01;//#并讲数据归一到0.01--1
			}
			index++;
			record = strtok(NULL, ",");
		}
	}
	label2list(label, target_label, OUTPUTNODES);
}

//计算各层的Sigmod激活函数输出
__global__ void Sigmod_kernel(float *A, int size)
{
	unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < size)
	{
		A[i] = 1 / (1 + exp(-A[i]));
	}
}

//向量减法 C=A-B
__global__ void VecSub_kernel(float *A, float *B, float *C, int size)
{
	unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < size)
	{
		C[i] = A[i] - B[i];
	}
}

//output_error*final_outputs*(1 - final_outputs)=output_error
//a=a*b*(1-b)
__global__ void argment1_kernel(float *a,float *b,int size)
{
	unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < size)
	{
		a[i]*=b[i] * (1 - b[i]);
	}
}

//C = ab'+C    a:row*1 b:col*1
//__global__ void updateW(float *a, float *b, float *C, unsigned int row, unsigned int col)
//{
//	float lr = LR;
//	unsigned int i = threadIdx.y + blockDim.y*blockIdx.y;
//	unsigned int j = threadIdx.x + blockDim.x*blockIdx.x;
//	if (i < row && j< col)
//	{
//		C[i + j*row] += lr*a[i] * b[j];//以列优先存储
//	}
//}

//训练神经网络，SGD
void train(float *d_Input,float *d_Hidden,float *d_Output,float *d_Target,float *d_Wih,float *d_Who,float *output_error,float *hidden_error)
{
	cublasHandle_t handle = 0;
	cublasCreate(&handle);
	float alpha = 1.0;
	float beta = 0.0;
	float alpha2 = LR;
	unsigned int offset_input = 0;
	unsigned int offset_output = 0;
	for (int i = 0; i < TRAIN_SIZE; i++)
	{
		offset_input = i*INPUTNODES;
		offset_output = i*OUTPUTNODES;

		//前向传播
		cublasSgemv(handle, CUBLAS_OP_N, HIDDENNODES, INPUTNODES, &alpha, d_Wih, HIDDENNODES, d_Input+ offset_input, 1, &beta, d_Hidden, 1);
		Sigmod_kernel << <1, HIDDENNODES >> > (d_Hidden, HIDDENNODES);
		cublasSgemv(handle, CUBLAS_OP_N, OUTPUTNODES, HIDDENNODES, &alpha, d_Who, OUTPUTNODES, d_Hidden, 1, &beta, d_Output, 1);
		Sigmod_kernel << <1, OUTPUTNODES >> > (d_Output, OUTPUTNODES);

		//误差反向传播，
		VecSub_kernel << <1, OUTPUTNODES >> > (d_Target+ offset_output, d_Output, output_error, OUTPUTNODES);
		cublasSgemv(handle, CUBLAS_OP_T, OUTPUTNODES, HIDDENNODES, &alpha, d_Who, OUTPUTNODES, output_error, 1, &beta, hidden_error, 1);

		//更新权值
		argment1_kernel << <1, OUTPUTNODES >> > (output_error, d_Output, OUTPUTNODES);
		cublasSger(handle, OUTPUTNODES, HIDDENNODES, &alpha2, output_error, 1, d_Hidden, 1, d_Who, OUTPUTNODES);
		argment1_kernel << <1, HIDDENNODES >> > (hidden_error, d_Hidden, HIDDENNODES);
		cublasSger(handle, HIDDENNODES, INPUTNODES, &alpha2, hidden_error, 1, d_Input + offset_input, 1, d_Wih, HIDDENNODES);
	}
	cublasDestroy(handle);
}

//使用训练后的神经网络进行预测
void query(float *d_Input, float *d_Hidden, float *d_Output,float *Output,float *d_Wih, float *d_Who,float *d_Target)
{
	cublasHandle_t handle = 0;
	cublasCreate(&handle);
	float alpha = 1.0;
	float beta = 0.0;
	unsigned int offset_input = 0;
	unsigned int offset_output = 0;
	for (int i = 0; i < TEST_SIZE; i++)
	{
		offset_input = i*INPUTNODES;
		offset_output = i*OUTPUTNODES;
		cublasSgemv(handle, CUBLAS_OP_N, HIDDENNODES, INPUTNODES, &alpha, d_Wih, HIDDENNODES, d_Input+ offset_input, 1, &beta, d_Hidden, 1);
		Sigmod_kernel << <1, HIDDENNODES >> > (d_Hidden, HIDDENNODES);
		cublasSgemv(handle, CUBLAS_OP_N, OUTPUTNODES, HIDDENNODES, &alpha, d_Who, OUTPUTNODES, d_Hidden, 1, &beta, d_Output, 1);
		Sigmod_kernel << <1, OUTPUTNODES >> > (d_Output, OUTPUTNODES);
		cudaMemcpy(d_Target+ offset_output, d_Output, sizeof(float)*OUTPUTNODES, cudaMemcpyDeviceToDevice);
	}
	cudaMemcpy(Output, d_Target, sizeof(float)*OUTPUTNODES*TEST_SIZE, cudaMemcpyDeviceToHost);//将全部测试结果传为主机
	cublasDestroy(handle);
}
int main(int argc, char **argv)
{
	// 启动设备
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s starting reduction at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	//分配主机网络空间
	float *Input,*Output,*all_Target,*Wih,*Who;

	Input = (float*)malloc(sizeof(float)*INPUTNODES*TRAIN_SIZE);
	//改为分配主机固定内存
	//CHECK(cudaMallocHost((void**)&Input, sizeof(float)*INPUTNODES));

	//获取全部结果
	Output = (float*)malloc(sizeof(float)*OUTPUTNODES*TEST_SIZE);

	//改为分配主机固定内存
	//CHECK(cudaMallocHost((void**)&Target, sizeof(float)*OUTPUTNODES));

	all_Target = (float*)malloc(sizeof(float)*OUTPUTNODES*TRAIN_SIZE);
	Wih = (float*)malloc(sizeof(float)*HIDDENNODES*INPUTNODES);
	Who = (float*)malloc(sizeof(float)*OUTPUTNODES*HIDDENNODES);
	
	//初始化权值在-0.5到0.5之间
	InitWeigth(Wih,Who);

	//分配设备网络
	float *d_Input, *d_Hidden, *d_Output, *d_Target, *d_Wih, *d_Who;

	//将数据一次性拷贝到显存中
	CHECK(cudaMalloc((float **)&d_Input, sizeof(float)*INPUTNODES*TRAIN_SIZE));
	CHECK(cudaMalloc((float **)&d_Target, sizeof(float)*OUTPUTNODES*TRAIN_SIZE));

	CHECK(cudaMalloc((float **)&d_Hidden, sizeof(float)*HIDDENNODES));
	CHECK(cudaMalloc((float **)&d_Output, sizeof(float)*OUTPUTNODES));


	float *output_error, *hidden_error;
	CHECK(cudaMalloc((float **)&output_error, sizeof(float)*OUTPUTNODES));
	CHECK(cudaMalloc((float **)&hidden_error, sizeof(float)*HIDDENNODES));

	//开辟权值显存空间，以列优先存储准则
	CHECK(cudaMalloc((void**)&d_Wih, sizeof(float)*HIDDENNODES*INPUTNODES));
	CHECK(cudaMalloc((void**)&d_Who, sizeof(float)*HIDDENNODES*OUTPUTNODES));
	//传递初始化的权值
	CHECK(cudaMemcpy(d_Wih, Wih, sizeof(float)*HIDDENNODES*INPUTNODES, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_Who, Who, sizeof(float)*HIDDENNODES*OUTPUTNODES, cudaMemcpyHostToDevice));


	//开辟设备权值二维空间显存
	//size_t pitch_wih, pitch_who;
	//CHECK(cudaMallocPitch((float **)&d_Wih, &pitch_wih, sizeof(float)*INPUTNODES, HIDDENNODES));
	//CHECK(cudaMallocPitch((float **)&d_Who, &pitch_who, sizeof(float)*HIDDENNODES, OUTPUTNODES));
	//
	//传递初始的权值到设备
	//CHECK(cudaMemcpy2D(d_Wih, pitch_wih, Wih,sizeof(float)*INPUTNODES, sizeof(float)*INPUTNODES, HIDDENNODES, cudaMemcpyHostToDevice));
	//CHECK(cudaMemcpy2D(d_Who, pitch_who, Who, sizeof(float)*HIDDENNODES, sizeof(float)*HIDDENNODES, OUTPUTNODES, cudaMemcpyHostToDevice));
	
	float time_start, time_used;

	//////////////打开训练集,开始训练////////////////////////////////
	FILE *fp = fopen("E:\\makemyownneural\\mnist_dataset\\mnist_train.csv", "at+");
	if (fp == NULL)
	{
		printf("训练集打开失败！\n");
		exit(EXIT_FAILURE);
	}
	printf("Training ......\n");
	time_start = seconds();
	//拷贝全部训练集数据。到显存
	unsigned int offset_input = 0;
	unsigned int offset_output = 0;
	for (int i = 0; i < TRAIN_SIZE; i++)
	{
		offset_input = i*INPUTNODES;
		offset_output = i*OUTPUTNODES;
		getData(fp, Input+ offset_input, all_Target+ offset_output);
		//CHECK(cudaMemcpy(&d_Input[offset_input], Input, sizeof(float)*INPUTNODES, cudaMemcpyHostToDevice));
		//CHECK(cudaMemcpy(&d_Target[offset_output], Target, sizeof(float)*OUTPUTNODES, cudaMemcpyHostToDevice));
	}
	CHECK(cudaMemcpy(d_Input, Input, sizeof(float)*INPUTNODES*TRAIN_SIZE, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_Target, all_Target, sizeof(float)*OUTPUTNODES*TRAIN_SIZE, cudaMemcpyHostToDevice));
	//训练
	train(d_Input, d_Hidden, d_Output, d_Target, d_Wih, d_Who, output_error, hidden_error);

	fclose(fp);//训练结束
	CHECK(cudaDeviceSynchronize());
	time_used = seconds() - time_start;
	printf("  training over ,time used %fs\n", time_used);


	///////////////开始测试/////////////////////////////////////////
	fp = fopen("E:\\makemyownneural\\mnist_dataset\\mnist_test.csv", "at+");
	if (fp == NULL)
	{
		printf("测试集打开失败！\n");
		exit(EXIT_FAILURE);
	}
	printf("Testing ......\n");
	time_start = seconds();
	//拷贝全部测试集数据。到显存
	for (int i = 0; i < TEST_SIZE; i++)
	{
		offset_input = i*INPUTNODES;
		offset_output = i*OUTPUTNODES;
		getData(fp, Input+offset_input, all_Target+offset_output);
		//CHECK(cudaMemcpy(&d_Input[offset_input], Input, sizeof(float)*INPUTNODES, cudaMemcpyHostToDevice));
		//CHECK(cudaMemcpy(&all_Target[offset_output], Target, sizeof(float)*OUTPUTNODES, cudaMemcpyHostToHost));
	}
	CHECK(cudaMemcpy(d_Input, Input, sizeof(float)*INPUTNODES*TEST_SIZE, cudaMemcpyHostToDevice));
	query(d_Input, d_Hidden, d_Output, Output, d_Wih, d_Who,d_Target);
	int current_count = 0;
	for (int i = 0; i < TEST_SIZE;i++)
	{
		offset_output = i*OUTPUTNODES;
		if (list2label(Output+offset_output, OUTPUTNODES) == list2label(all_Target+offset_output, OUTPUTNODES))
		{
			current_count++;
		}
	}
	fclose(fp);//训练结束

	////////////////训练结果////////////////////////////////////////////////////////
	CHECK(cudaDeviceSynchronize());
	time_used = seconds() - time_start;
	printf("  test over,time used %fs\n", time_used);
	printf("Consequenction:\n");
	printf("  currectCount= %d total_count=%d\n", current_count, TEST_SIZE);
	printf("  the total accuracy is %f\n", (float)(current_count) / TEST_SIZE);

	//CHECK(cudaMemcpy2D(test_Wih, sizeof(float)*INPUTNODES,d_Wih, pitch_wih, sizeof(float)*INPUTNODES, HIDDENNODES, cudaMemcpyDeviceToHost));
	//CHECK(cudaMemcpy2D(test_Who, sizeof(float)*HIDDENNODES,d_Who, pitch_who, sizeof(float)*HIDDENNODES, OUTPUTNODES, cudaMemcpyDeviceToHost));
	


	//释放主机内存
	free(Input);
	free(Output);
	free(all_Target);
	free(Wih);
	free(Who);
	
	//释放设备内存
	CHECK(cudaFree(d_Input));
	CHECK(cudaFree(d_Hidden));
	CHECK(cudaFree(d_Output));
	CHECK(cudaFree(d_Target));
	CHECK(cudaFree(d_Wih));
	CHECK(cudaFree(d_Who));
	CHECK(cudaFree(output_error));
	CHECK(cudaFree(hidden_error));
	getchar();
	return 0;
}
