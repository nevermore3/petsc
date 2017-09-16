#include "type.h"

void C_Init(SPMultiply *SPMul, SparseMatrixLoc_mumps *C_loc, Info *info, int argc, char *argv[]);
void Eig(int argc, char **argv,SPMultiply SPMul, SparseMatrixLoc_mumps C_loc, Info info);


int main(int argc, char *argv[])
{
	int myid, np;
    double mytime1, mytime2;
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    /**************  Init the program   *******************/
    Info info;        
    SPMultiply SPMul;      //稀疏矩阵相乘
    SparseMatrixLoc_mumps C_loc;     //for  what  
    mytime1 = MPI_Wtime();
    C_Init(&SPMul, &C_loc, &info, argc, argv);
    mytime2 = MPI_Wtime();
	if(myid == 0)
    	printf("\n\t Init time %lf \n\n", mytime2 - mytime1);

    /**************  test the program   *******************/
    MPI_Barrier(MPI_COMM_WORLD);

    /**************  set the parameters ********************/
    info.nGau = 4;
    info.M0 = 19;        //投影子空间的s大小

    /*****************特征值的求解范围************************/
	//100-0.010756 110-0.0089  20-0.257405  50-0.041931  10-0.946389 200-0.00276   130-0.00657  160-0.00456  180-0.00354
	//60-0.031207   70-0.21660  80-0.016173 90-0.013890
    info.Emax =  0.010756;   
    info.Emin= 0.00;  


    mytime1 = MPI_Wtime();
    /**************  eig solve program   *******************/
    Eig(argc, argv,SPMul, C_loc, info);

    mytime2 = MPI_Wtime();
	if(myid == 0)
    	printf("\n\t Eig time %lf \n\n", mytime2 - mytime1);

	/****************free the store**************************/
	/*
	free(SPMul.ia_in);
	free(SPMul.ja_in);
	free(SPMul.a_in);
	free(SPMul.ia_out);
	free(SPMul.ja_out);
	free(SPMul.a_out);
	free(SPMul.recv_rowid);
	free(SPMul.recv_count);
	free(SPMul.send_rowid);
	free(SPMul.send_count);
	free(C_loc.c_loc);
	free(C_loc.c_diag);
	free(C_loc.rhs);
	free(C_loc.ic_loc);
	free(C_loc.jc_loc);
	free(info.N_loc);
	free(info.fstrow);

	*/
    /**************  end the program   *******************/
    MPI_Finalize();
    return 0;
}
