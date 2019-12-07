///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// version 3.0
// title: ����cuda����������д����ʶ����Ż�
// �㷨��SGD
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
#define INPUTNODES 784   //�����ڵ���
#define HIDDENNODES 200  //���ز�ڵ���
#define OUTPUTNODES 10   //�����ڵ���
#define TRAIN_SIZE 60000 //ѵ����������
#define TEST_SIZE 10000  //���Լ�������
#define LR 0.1           //ѧϰ��
//��ȡʱ��,��λ��
inline double seconds()
{
	LARGE_INTEGER nFreq;//LARGE_INTEGER��64λϵͳ����LONGLONG����32λϵͳ���Ǹߵ�����32λ��LONG����windows.h��ͨ��Ԥ�����������
	LARGE_INTEGER nTime;//��¼��ʼʱ�ļ�������ֵ
	double time;

	QueryPerformanceFrequency(&nFreq);//��ȡϵͳʱ��Ƶ��
	QueryPerformanceCounter(&nTime);//��ȡ��ʼʱ�̼���ֵ
	time = (double)(nTime.QuadPart) / (double)nFreq.QuadPart;
	return time;
}
//struct NeuralNetwork {
//	float *Input; //[INPUTNODES]//���������
//	float *Hidden;//[HIDDENNODES]//���ز����
//	float *Output;//[OUTPUTNODES]//��������
//	float *Target;//[OUTPUTNODES]//Ŀ�����
//	float *Wih;//[HIDDENNODES][INPUTNODES]//����㵽���ز�Ȩֵ
//	float *Who;//[OUTPUTNODES][HIDDENNODES]//���ز㵽�����Ȩֵ
//};

//�����ʼ��Ȩֵ��-0.5��0.5֮��
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

//////���ݴ���//////////////
//Ŀ�������� one-hot
// label=1����,list����Ϊ[0,1 0.99 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
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

//one-hotת��Ϊ����label
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

//��ȡͼƬ���ݵ�inputImage�У�ͼƬĿ��������target_label
void getData(FILE *fp, float *inputImage, float *target_label)
{
	int label = 0;
	char *line, *record;
	char buffer[2048];
	if ((line = fgets(buffer, sizeof(buffer), fp)) != NULL)//��û�ж�ȡ���ļ�ĩβʱ����
	{
		record = strtok(line, ",");
		int index = 0;
		while (record != NULL)//��ȡÿһ�е�����
		{
			if (index == 0)
			{
				label = atoi(record);
			}
			else
			{
				inputImage[index - 1] = ((float)(atoi(record)) / 255.0) * 0.99 + 0.01;//#�������ݹ�һ��0.01--1
			}
			index++;
			record = strtok(NULL, ",");
		}
	}
	label2list(label, target_label, OUTPUTNODES);
}

//��������Sigmod��������
__global__ void Sigmod_kernel(float *A, int size)
{
	unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < size)
	{
		A[i] = 1 / (1 + exp(-A[i]));
	}
}

//�������� C=A-B
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
//		C[i + j*row] += lr*a[i] * b[j];//�������ȴ洢
//	}
//}

//ѵ�������磬SGD
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

		//ǰ�򴫲�
		cublasSgemv(handle, CUBLAS_OP_N, HIDDENNODES, INPUTNODES, &alpha, d_Wih, HIDDENNODES, d_Input+ offset_input, 1, &beta, d_Hidden, 1);
		Sigmod_kernel << <1, HIDDENNODES >> > (d_Hidden, HIDDENNODES);
		cublasSgemv(handle, CUBLAS_OP_N, OUTPUTNODES, HIDDENNODES, &alpha, d_Who, OUTPUTNODES, d_Hidden, 1, &beta, d_Output, 1);
		Sigmod_kernel << <1, OUTPUTNODES >> > (d_Output, OUTPUTNODES);

		//���򴫲���
		VecSub_kernel << <1, OUTPUTNODES >> > (d_Target+ offset_output, d_Output, output_error, OUTPUTNODES);
		cublasSgemv(handle, CUBLAS_OP_T, OUTPUTNODES, HIDDENNODES, &alpha, d_Who, OUTPUTNODES, output_error, 1, &beta, hidden_error, 1);

		//����Ȩֵ
		argment1_kernel << <1, OUTPUTNODES >> > (output_error, d_Output, OUTPUTNODES);
		cublasSger(handle, OUTPUTNODES, HIDDENNODES, &alpha2, output_error, 1, d_Hidden, 1, d_Who, OUTPUTNODES);
		argment1_kernel << <1, HIDDENNODES >> > (hidden_error, d_Hidden, HIDDENNODES);
		cublasSger(handle, HIDDENNODES, INPUTNODES, &alpha2, hidden_error, 1, d_Input + offset_input, 1, d_Wih, HIDDENNODES);
	}
	cublasDestroy(handle);
}

//ʹ��ѵ��������������Ԥ��
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
	cudaMemcpy(Output, d_Target, sizeof(float)*OUTPUTNODES*TEST_SIZE, cudaMemcpyDeviceToHost);//��ȫ�����Խ����Ϊ����
	cublasDestroy(handle);
}
int main(int argc, char **argv)
{
	// �����豸
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s starting reduction at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	//������������ռ�
	float *Input,*Output,*all_Target,*Wih,*Who;

	Input = (float*)malloc(sizeof(float)*INPUTNODES*TRAIN_SIZE);
	//��Ϊ���������̶��ڴ�
	//CHECK(cudaMallocHost((void**)&Input, sizeof(float)*INPUTNODES));

	//��ȡȫ�����
	Output = (float*)malloc(sizeof(float)*OUTPUTNODES*TEST_SIZE);

	//��Ϊ���������̶��ڴ�
	//CHECK(cudaMallocHost((void**)&Target, sizeof(float)*OUTPUTNODES));

	all_Target = (float*)malloc(sizeof(float)*OUTPUTNODES*TRAIN_SIZE);
	Wih = (float*)malloc(sizeof(float)*HIDDENNODES*INPUTNODES);
	Who = (float*)malloc(sizeof(float)*OUTPUTNODES*HIDDENNODES);
	
	//��ʼ��Ȩֵ��-0.5��0.5֮��
	InitWeigth(Wih,Who);

	//�����豸����
	float *d_Input, *d_Hidden, *d_Output, *d_Target, *d_Wih, *d_Who;

	//������һ���Կ������Դ���
	CHECK(cudaMalloc((float **)&d_Input, sizeof(float)*INPUTNODES*TRAIN_SIZE));
	CHECK(cudaMalloc((float **)&d_Target, sizeof(float)*OUTPUTNODES*TRAIN_SIZE));

	CHECK(cudaMalloc((float **)&d_Hidden, sizeof(float)*HIDDENNODES));
	CHECK(cudaMalloc((float **)&d_Output, sizeof(float)*OUTPUTNODES));


	float *output_error, *hidden_error;
	CHECK(cudaMalloc((float **)&output_error, sizeof(float)*OUTPUTNODES));
	CHECK(cudaMalloc((float **)&hidden_error, sizeof(float)*HIDDENNODES));

	//����Ȩֵ�Դ�ռ䣬�������ȴ洢׼��
	CHECK(cudaMalloc((void**)&d_Wih, sizeof(float)*HIDDENNODES*INPUTNODES));
	CHECK(cudaMalloc((void**)&d_Who, sizeof(float)*HIDDENNODES*OUTPUTNODES));
	//���ݳ�ʼ����Ȩֵ
	CHECK(cudaMemcpy(d_Wih, Wih, sizeof(float)*HIDDENNODES*INPUTNODES, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_Who, Who, sizeof(float)*HIDDENNODES*OUTPUTNODES, cudaMemcpyHostToDevice));


	//�����豸Ȩֵ��ά�ռ��Դ�
	//size_t pitch_wih, pitch_who;
	//CHECK(cudaMallocPitch((float **)&d_Wih, &pitch_wih, sizeof(float)*INPUTNODES, HIDDENNODES));
	//CHECK(cudaMallocPitch((float **)&d_Who, &pitch_who, sizeof(float)*HIDDENNODES, OUTPUTNODES));
	//
	//���ݳ�ʼ��Ȩֵ���豸
	//CHECK(cudaMemcpy2D(d_Wih, pitch_wih, Wih,sizeof(float)*INPUTNODES, sizeof(float)*INPUTNODES, HIDDENNODES, cudaMemcpyHostToDevice));
	//CHECK(cudaMemcpy2D(d_Who, pitch_who, Who, sizeof(float)*HIDDENNODES, sizeof(float)*HIDDENNODES, OUTPUTNODES, cudaMemcpyHostToDevice));
	
	float time_start, time_used;

	//////////////��ѵ����,��ʼѵ��////////////////////////////////
	FILE *fp = fopen("E:\\makemyownneural\\mnist_dataset\\mnist_train.csv", "at+");
	if (fp == NULL)
	{
		printf("ѵ������ʧ�ܣ�\n");
		exit(EXIT_FAILURE);
	}
	printf("Training ......\n");
	time_start = seconds();
	//����ȫ��ѵ�������ݡ����Դ�
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
	//ѵ��
	train(d_Input, d_Hidden, d_Output, d_Target, d_Wih, d_Who, output_error, hidden_error);

	fclose(fp);//ѵ������
	CHECK(cudaDeviceSynchronize());
	time_used = seconds() - time_start;
	printf("  training over ,time used %fs\n", time_used);


	///////////////��ʼ����/////////////////////////////////////////
	fp = fopen("E:\\makemyownneural\\mnist_dataset\\mnist_test.csv", "at+");
	if (fp == NULL)
	{
		printf("���Լ���ʧ�ܣ�\n");
		exit(EXIT_FAILURE);
	}
	printf("Testing ......\n");
	time_start = seconds();
	//����ȫ�����Լ����ݡ����Դ�
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
	fclose(fp);//ѵ������

	////////////////ѵ�����////////////////////////////////////////////////////////
	CHECK(cudaDeviceSynchronize());
	time_used = seconds() - time_start;
	printf("  test over,time used %fs\n", time_used);
	printf("Consequenction:\n");
	printf("  currectCount= %d total_count=%d\n", current_count, TEST_SIZE);
	printf("  the total accuracy is %f\n", (float)(current_count) / TEST_SIZE);

	//CHECK(cudaMemcpy2D(test_Wih, sizeof(float)*INPUTNODES,d_Wih, pitch_wih, sizeof(float)*INPUTNODES, HIDDENNODES, cudaMemcpyDeviceToHost));
	//CHECK(cudaMemcpy2D(test_Who, sizeof(float)*HIDDENNODES,d_Who, pitch_who, sizeof(float)*HIDDENNODES, OUTPUTNODES, cudaMemcpyDeviceToHost));
	


	//�ͷ������ڴ�
	free(Input);
	free(Output);
	free(all_Target);
	free(Wih);
	free(Who);
	
	//�ͷ��豸�ڴ�
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
