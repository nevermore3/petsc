#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef struct DCOMPLEX
{
    double i;
    double r;
}dcomplex;

typedef struct SPARSEMATRIX  //稀疏矩阵结构
{
    int *ia;   
    int *ja;  
    double *a;
    int N;  //多少行
    int nnz;   //非零元素的个数
} SparseMatrix;


typedef struct SPARSEMATRIXLOC   //稀疏矩阵在各个处理器上的数据结构  行压缩存储格式
{
	int *in;
	int *out;
    int *ia_loc;      
    int *ja_loc;
    double *a_loc;
	int N_loc;           //每个进程的处理的行数
	int nnz_loc;         //每个进程处理的非零元的个数
    int fstrow;          //相对于全局矩阵每个进程处理行数的偏置值
} SparseMatrixLoc;


typedef struct SPARSEMATRIXMULTIPLY   //稀疏矩阵相乘的数据结构
{
    int M0;      //子空间大小
    int fstrow;  //偏置值
    int N_loc;
    /************相乘时候 内点*************/
    int nnz_in;
    int *ia_in;     
    int *ja_in;
    double *a_in;
    /************相乘时候 外点**************/
    int nnz_out;
    int *ia_out;
    int *ja_out;
    double *a_out;
    

    double *B_out;

    int nrecv;
    int *recv_rowid;
    int *recv_count;

    int nsend;
    int *send_rowid;
    int *send_count;
} SPMultiply;


typedef struct INFORMATION
{
    

    int *N_loc;
    int N;          //矩阵行数
    int nGau;       //高斯点个数 
    int M0;         //投影空间的大小

    double Emax;
    double Emin;
    

    int times;

    int myid;      //进程ID
    int np;        //进程总数
    int *fstrow;

} Info;


typedef struct SPARSEMATRIXLOC_MUMPS
{
    dcomplex *c_loc;
    double *c_diag;
    dcomplex *rhs;    //mumps计算线性系统的右端项  

    int N;
    int nnz_loc;
    int N_loc;

    int *ic_loc;      //三元组形式的行
    int *jc_loc;       //三元组形式的列
	int max_in;
	int max_out;
    int  nrhs;        
	int *in;
	int *out;
}SparseMatrixLoc_mumps;



