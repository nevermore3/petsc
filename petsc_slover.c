#include <petscksp.h>
#include "type.h"


static char help[] = "Solves a linear system in parallel with ksp.\n\n";
void linersystem_slover(int argc, char **args, Info *info, SparseMatrixLoc_mumps *C_loc, dcomplex *rhs)
{
    Mat    A;
    Vec    x,b, b2;      //x:近似解  b：右端项
    KSP    ksp;
    PetscInt  myid, its, i,j,k,n,N,M0,nnz,fstrow,index,np,nlocal;
    PetscInt    *ia, *ja, nrm;
    PetscScalar   value;   
    PetscErrorCode  ierr;
	PetscScalar    *xa;
	dcomplex *a, *answer;
	double time1, time2;
    PetscInitialize(&argc, &args,(char*)0, help);
    #if !defined(PETSC_USE_COMPLEX)
        SETERRQ(MPI_COMM_WORLD, 1, "This code requires complex numbers");
    #endif
	myid = (*info).myid;
    N  = (*info).N;
    M0 = (*info).M0;
    ia = (*C_loc).ic_loc;
    ja = (*C_loc).jc_loc;
    a  = (*C_loc).c_loc;
	nnz= (*C_loc).nnz_loc;
	np = (*info).np;
	fstrow = (*info).fstrow[myid];
	int N_loc = (*info).N_loc[myid];

	MPI_Datatype  MPI_complex;
	MPI_Type_contiguous(2,MPI_DOUBLE, &MPI_complex);
	MPI_Type_commit(&MPI_complex);
    /*************************set the matrix***********************************************/
	time1 = MPI_Wtime();
	ierr = MatCreate(PETSC_COMM_WORLD, &A);          CHKERRV(ierr);
	ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N); CHKERRV(ierr);
	ierr = MatSetType(A, MATMPIAIJ);                 CHKERRV(ierr);
    ierr = MatMPIAIJSetPreallocation(A,0,(*C_loc).in , 0,(*C_loc).out ); CHKERRV(ierr);
	ierr = MatSetUp(A);                             CHKERRV(ierr);
	time2 = MPI_Wtime();
//	if(myid == 0)
//		printf("Matrix SetUp time is %lf\n", time2-time1);
	//ierr =MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); CHKERRQ(ierr);
//	ierr= MatSetValues(A,(*C_loc).N_loc, ia,N,ja,a, INSERT_VALUES);  CHKERRQ(ierr);
    
	time1 = MPI_Wtime();
	for (i = 0; i < nnz; i++)
	{
		value = a[i].r+a[i].i*PETSC_i;
		ierr = MatSetValue(A, ia[i], ja[i] , value , INSERT_VALUES); CHKERRV(ierr);
	}
	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);  CHKERRV(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);    CHKERRV(ierr);
	time2 = MPI_Wtime();
//	if(myid == 0)
//		printf("MatrixSetValue time is %lf\n", time2 - time1);
//	ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);	CHKERRV(ierr);
	/************************************************************************************/


	ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRV(ierr);
	ierr = VecSetSizes(b, PETSC_DECIDE, N); CHKERRV(ierr);
	ierr = VecSetType(b, VECMPI);           CHKERRV(ierr);
	ierr = VecDuplicate(b, &x);             CHKERRV(ierr);

	time1 = MPI_Wtime();
    for(i = 0; i < M0; i++)
    {
       	for(j = 0; j < N_loc ; j++)
    	{
    		index = fstrow + j;
	//		printf("pid is %d\tindex is %d\n",myid,index);
    	//	value = rhs[j * M0 + i].r +rhs[j *M0 + i].i * PETSC_i;
    		value = rhs[j +i * N_loc].r +rhs[j  + i*N_loc].i * PETSC_i;
    		ierr = VecSetValue(b, index, value, INSERT_VALUES);		CHKERRV(ierr);
    	}
    	ierr = VecAssemblyBegin(b);          CHKERRV(ierr);
		ierr = VecAssemblyEnd(b);            CHKERRV(ierr);
//		ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);  CHKERRV(ierr);
  
        ierr=KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRV(ierr);
        ierr=KSPSetOperators(ksp,A,A);       CHKERRV(ierr);
	//	ierr=KSPSetTolerances(ksp,1.e-9,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRV(ierr);
        ierr=KSPSetFromOptions(ksp);         CHKERRV(ierr);
        ierr=KSPSolve(ksp,b,x);              CHKERRV(ierr);
//		ierr=VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRV(ierr);
		ierr = KSPGetIterationNumber(ksp, &its);     CHKERRV(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Iteration is %d\n", its);   CHKERRV(ierr);
        /*组装求解出来的结果*/
		ierr = VecGetLocalSize(x, &nlocal);         CHKERRV(ierr);
		ierr =VecGetArray(x,&xa);   CHKERRV(ierr);

		for( j = 0; j < N_loc; j++)
		{
			rhs[j + i * N_loc].r = (double)PetscRealPart(xa[j]);
			rhs[j +i * N_loc].i = (double)PetscImaginaryPart(xa[j]);
		}
		
    }
	time2 = MPI_Wtime();
//	if(myid == 0)
//		printf("KSP time is %lf\n", time2 - time1);
    /************************************************************************
                    Free work space
    ************************************************************************/
	ierr = VecRestoreArray(x,&xa);    		 CHKERRV(ierr);
    ierr=KSPDestroy(&ksp);                   CHKERRV(ierr);
    ierr=VecDestroy(&x);                     CHKERRV(ierr);
    ierr=VecDestroy(&b);                     CHKERRV(ierr);
    ierr=MatDestroy(&A);                     CHKERRV(ierr);
//	ierr=VecDestroy(&b2);                    CHKERRV(ierr);
    ierr=PetscFinalize();                    CHKERRV(ierr);

}
