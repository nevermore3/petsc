#include "type.h"
#include <stdlib.h>
#include <ctype.h>

void Feast_SetRandomMatrix(double *Q_loc, int length, int myid);
void Feast_GetProjectionOp(int argc, char **argv,double *Y_loc, double *Q_loc, SparseMatrixLoc_mumps C_loc,  Info info);
void Feast_Project(double *Q_loc, double *lambda, double *Y_loc, SPMultiply SPMul, Info info);
void Feast_GetRes(double *maxres, double *lambda, double  *Q_loc, SPMultiply SPMul, Info info);
void orth(double *Q_loc, double *Y_loc, int N_loc, int M0);
void linersystem_slover(int argc, char **args, Info *info, SparseMatrixLoc_mumps *C_loc ,dcomplex *rhs); 
void Eig(int argc, char **argv, SPMultiply SPMul, SparseMatrixLoc_mumps C_loc, Info info)   
{
    int i=0, j,k;
	double time1, time2;
    int myid = info.myid;
    int M0 = info.M0;
    int N_loc = info.N_loc[myid];   
    C_loc.nrhs = M0;  
    int length = M0 * N_loc;
    double *Y_loc = calloc(length, sizeof(double));
    double *Q_loc = calloc(length, sizeof(double));
    Feast_SetRandomMatrix(Q_loc, length, myid);
    /********************  comput  **************************************/
    double maxres=1;
    double *lambda = calloc(M0, sizeof(double));  
    while(maxres>0.0001&& i<50)    
    {
	   time1 = MPI_Wtime();
       Feast_GetProjectionOp(argc,argv,Y_loc, Q_loc, C_loc, info);
	   time2 = MPI_Wtime();
	  	
	   if(myid == 0 )
	   {
	   		printf("Projection time is %lf \n", time2- time1);
	   }

	   time1 = MPI_Wtime();
       Feast_Project(Q_loc,lambda, Y_loc, SPMul, info);
	   time2 = MPI_Wtime();

	   if(myid == 0 )
	   {
	   	 	printf("Compute eigenvalue time is %lf\n", time2 - time1);
	   }

	   time1 = MPI_Wtime();
       Feast_GetRes(&maxres, lambda, Q_loc, SPMul, info);
	   time2 = MPI_Wtime();

	   if(myid == 0 )
	   {
	   		printf("Getres time is %lf\n", time2 - time1);
	   }
	   i++;
    }
	free(Y_loc);
	free(Q_loc);
}

/************************************
生成随机矩阵
*****************************************/
void Feast_SetRandomMatrix(double *Y_loc, int length, int myid)
{
    int three = 3;
    int iseed[4] = {myid, myid + 3, myid + 9, 1 };
    dlarnv_(&three, iseed, &length, Y_loc);   
}


void C_Add(SparseMatrixLoc_mumps C_loc, dcomplex Ze);
void Gauss(int nGauss, int i, double *xe, double *we);

/**************************通过围道积分计算投影矩阵**********************
********************************************************************/

void Feast_GetProjectionOp(int argc, char **argv, double *Y_loc, double *Q_loc, SparseMatrixLoc_mumps C_loc,  Info info)
{
    int myid = info.myid;
    int N = info.N;    //矩阵的大小
    int M0 = info.M0;      //投影子空间的大小
    int *N_loc = info.N_loc;  //每个进程处理多少行矩阵
    double Emax = info.Emax;
    double Emin = info.Emin;
    int i, j, k, nrow,id;
    int myN_loc = N_loc[myid];
	double time1, time2;
    /******************************收集分布存储的正交矩阵*******************************/
	if(myN_loc != C_loc.N_loc)
		printf("Error !!!!!!!!!!!!! myN_loc : %d != C_locN_loc : %d\n", myN_loc, C_loc.N_loc);
   
    dcomplex *rhs;
    rhs = calloc(myN_loc * M0, sizeof(dcomplex));
	memset(Y_loc, 0, sizeof(double)*myN_loc*M0);
    dcomplex Ze, jac;
    double Emid;
    Emid = 0.5 * (Emax + Emin);
    double r;
    r = (Emax - Emin)*0.5;
    for (i = 0; i < info.nGau; i++)
    {
        for(j = 0; j< myN_loc * M0; j++)
        {
            rhs[j].r = Q_loc[j];
            rhs[j].i = 0;
        }

        //复数矩阵加减（ZeI-A）
        double xe, we, theta;
        Gauss(info.nGau,  i, &xe,  &we);
        theta = -3.1415926535898 * (xe-1) *0.5;  
        Ze.r = Emid+  r * cos(theta);
        Ze.i = r * sin(theta);
	//	C_Add(C_loc, Ze);
       for(j = 0; j < C_loc.nnz_loc; j++)
       {
           if(C_loc.ic_loc[j] == C_loc.jc_loc[j])
           {
               id = C_loc.ic_loc[j] - C_loc.ic_loc[0];
               C_loc.c_loc[j].r = Ze.r - C_loc.c_diag[id];
               C_loc.c_loc[j].i = Ze.i;
           }
       }
		
		time1 = MPI_Wtime();
        linersystem_slover(argc, argv,&info, &C_loc, rhs);
		time2 = MPI_Wtime();
		
		if(myid == 0 && i == 0)
			printf("Linersystem is %lf\n", time2 - time1);

        jac.r = 0.25 * we * r  * cos(theta);
        jac.i = 0.25 * we * r  * sin(theta);
  //      for(j=0; j< 30; j++)
	//		printf("rhs[%d].r is %lf\t rsh[%d].i is %lf\n", j, rhs[j].r,j, rhs[j].i);
        for (j = 0; j <myN_loc * M0; j++)
        {
            Y_loc[j] = Y_loc[j] + jac.r * rhs[j].r - jac.i * rhs[j].i ;
        }
//		for(j=0; j< 30; j++)
//			printf("Y[%d] is %lf\n", j, Y_loc[j]);
    }
    free(rhs);
}


/************************************
根据计算得到的投影矩阵进行投影,并且将投影矩阵正交化
*****************************************/
void mul(double *result, double *Y_loc, SPMultiply SPMul, Info info);
void Feast_Project(double *Q_loc, double *lambda, double *Y_loc, SPMultiply SPMul, Info info)
{
   

    int myid = info.myid;
    int M0 = info.M0;
    int i, j;
    int N_loc = info.N_loc[myid];
    char char_N = 'N', char_T = 'T', char_V = 'V', char_U = 'U';
    double   one, zero;
    one = 1; 
    zero = 0; 
    int int_one = 1;
	
//		for(i = 0; i < 30; i++)
//		{
//			printf("Y_loc[%d] is %lf\t Q_loc[%d] is %lf\n",i,Y_loc[i],i,Q_loc[i]);
//		}
    /****************************** 投影Aq=YT*A*Y 和Bq=YT*Y *******************************/
    int error, lwork = N_loc * M0;
    double *Aq_loc, *Aq, *Bq_loc, *Bq, *work_loc;
    Aq_loc = calloc(M0 * M0, sizeof(double));           
    Bq_loc = calloc(M0 * M0, sizeof(double));
    work_loc = calloc(lwork, sizeof(double));
    Aq = calloc(M0 * M0, sizeof(double));
    Bq = calloc(M0 * M0, sizeof(double));

    //Aq=Y^t A Y
    mul(work_loc, Y_loc, SPMul, info);
    dgemm_(&char_T, &char_N, &M0, &M0, &N_loc, &one, Y_loc, &N_loc, work_loc, &N_loc, &zero, Aq_loc, &M0); //Aq = Y^T * work
    MPI_Reduce(Aq_loc, Aq,  M0 * M0, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    //Bq=Y^T * Y
    dgemm_(&char_T, &char_N, &M0, &M0, &N_loc, &one, Y_loc, &N_loc, Y_loc, &N_loc, &zero, Bq_loc, &M0);
    MPI_Reduce(Bq_loc, Bq,  M0 * M0, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	

    /****************************** 计算Rize对Aq*X=λ*Bq*X *******************************/
    if (myid == 0)
    {
//		for(i = 0; i < M0*M0; i++)
//		{
//			printf("Aq[%d] is %lf\t Bq[%d] is %lf\n",i,Aq[i],i,Bq[i]);
//		}
    	 dsygv_(&int_one, &char_V, &char_U, &M0,Aq,&M0,Bq,&M0, lambda,work_loc, &lwork, &error);
    	 if(error != 0)
	    	printf("!!!!!!!!!!error in dsygv_ !!!!!%d\n", error);

    }

	MPI_Bcast(Aq,M0*M0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	dgemm_(&char_N, &char_N, &N_loc, &M0, &M0, &one, Y_loc, &N_loc, Aq, &M0, &zero, Q_loc,&N_loc);
    free(work_loc);
    free(Aq_loc);
    free(Bq_loc);
    free(Aq);
    free(Bq);

}


void Feast_GetRes(double *maxres, double *lambda, double *Q_loc, SPMultiply SPMul, Info info)
{

    int myid = info.myid;
    int M0 = info.M0;
    int N_loc = info.N_loc[myid];

	double Emax = info.Emax;
	double Emin = info.Emin;


    int i, j, k;
    double  *work_loc, *res_loc, *res, *x_loc, *x;
    work_loc = calloc(N_loc * M0, sizeof(double));
    res_loc = calloc(M0, sizeof(double));
    x_loc = calloc(M0, sizeof(double));
    res = calloc(M0, sizeof(double));
    x = calloc(M0, sizeof(double));


    mul(work_loc, Q_loc, SPMul, info);
    MPI_Bcast(lambda, M0, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (i = 0; i < M0; i++)
    {
        for (j = 0; j < N_loc; j++)
        {
            work_loc[i * N_loc + j] =  work_loc[i * N_loc + j] - lambda[i]* Q_loc[i * N_loc + j] ;
        
        }
    }

    for (i = 0; i < M0; i++)
    {
        res_loc[i] = 0;
        x_loc[i] = 0;
        for (j = 0; j < N_loc; j++)
        {
		      res_loc[i] = res_loc[i] + fabs(work_loc[i*N_loc+j]);
			  x_loc[i]   = x_loc[i] + fabs(Q_loc[i*N_loc+j]);
        }
    }

    MPI_Reduce(res_loc, res, M0, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(x_loc,   x,   M0, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


    if (myid == 0)
    {
        for (i = 0; i < M0; i++)
            res[i] = res[i] / x[i];

        int neig = 0;
        int maxid;
        *maxres = 0;
        for (i = 0; i < M0; i++)
        {
            if ( lambda[i]< Emax && lambda[i] > Emin)
            {
                neig++;
                if (res[i] > *maxres)
                {
                    maxid = i;
                    *maxres = res[i];
                }
				
			printf("lambda %d is %.25f\n",i, lambda[i]);
            }
        }

     printf("\t\tcorect  maxres: %.25f\n", *maxres);

    }
    free(work_loc);
    free(res);
    free(x);
    free(res_loc);
    free(x_loc);

    MPI_Bcast(maxres, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

}




/********************************************二级程序**********************************/

void C_Add(SparseMatrixLoc_mumps C_loc, dcomplex Ze)
{
    int nnz_loc = C_loc.nnz_loc;
    int *ic_loc = C_loc.ic_loc;
    int *jc_loc = C_loc.jc_loc;
    int id;
    dcomplex *c_loc = C_loc.c_loc;
    double *c_diag = C_loc.c_diag;
    int i;
    for (i = 0; i < nnz_loc; i++)
    {
        if (ic_loc[i] == jc_loc[i])
        {
			id=ic_loc[i]-ic_loc[0];
            c_loc[i].r = Ze.r - c_diag[id];
            c_loc[i].i = Ze.i;
        }
    }

}

/************************************
矩阵乘
*****************************************/
void mul(double *result, double *Y_loc, SPMultiply SPMul, Info info)
{
    int M0 = info.M0;
    int np = info.np;
    int myid = info.myid;


    int N_loc = SPMul.N_loc;

    int nnz_in = SPMul.nnz_in;
    int *ia_in = SPMul.ia_in;
    int *ja_in = SPMul.ja_in;
    double *a_in = SPMul.a_in;

    int nnz_out = SPMul.nnz_out;
    int *ia_out = SPMul.ia_out;
    int *ja_out = SPMul.ja_out;
    double *a_out = SPMul.a_out;

    int nsend = SPMul.nsend;
    int nrecv = SPMul.nrecv;
    int *recv_rowid = SPMul.recv_rowid;
    int *recv_count = SPMul.recv_count;
    int *send_rowid = SPMul.send_rowid;
    int *send_count = SPMul.send_count;

    MPI_Status status;

    int i, j;

    double *B_out;
    if (nrecv == 0)
        B_out = calloc(1, sizeof(double));
    else
        B_out = calloc(nrecv * M0, sizeof(double));

    int rowid, colid, idsend, idrecv, sendoffset, recvoffset, number, mpitag = 0;

    double *sendbuf;
    sendbuf = calloc(N_loc * M0, sizeof(double));
    idsend = 0;
    idrecv = 0;
    recvoffset = 0;
    sendoffset = 0;

    while (recv_count[idsend] == 0 && idsend < np)
        idsend++;
    while (send_count[idrecv] == 0 && idrecv < np)
        idrecv++;
    while (idsend < np || idrecv < np)
    {
        if (idsend == idrecv)
        {
            if (idrecv < myid)
            {
                MPI_Recv(B_out + recvoffset, recv_count[idrecv] * M0, MPI_DOUBLE, idrecv, mpitag, MPI_COMM_WORLD, &status);
                recvoffset = recvoffset + recv_count[idrecv] * M0;
                idrecv++;
            }

            if (idsend > myid)
            {
                for (i = 0; i < send_count[idsend]; i++)
                {
                    rowid = send_rowid[sendoffset];
                    for (j = 0; j < M0; j++)
                        sendbuf[i * M0 + j] = Y_loc[rowid + j * N_loc];
                    sendoffset++;
                }
                MPI_Send(sendbuf, send_count[idsend] * M0, MPI_DOUBLE, idsend, mpitag, MPI_COMM_WORLD);
                idsend++;
            }
        }
        else
        {
            if (idrecv < idsend || idsend == np)
            {
                MPI_Recv(B_out + recvoffset, recv_count[idrecv] * M0, MPI_DOUBLE, idrecv, mpitag, MPI_COMM_WORLD, &status);
                recvoffset = recvoffset + recv_count[idrecv] * M0;
                idrecv++;
            }
            else if (idsend < idrecv || idrecv == np)
            {
                for (i = 0; i < send_count[idsend]; i++)
                {
                    rowid = send_rowid[sendoffset];
                    for (j = 0; j < M0; j++)
                        sendbuf[i * M0 + j] = Y_loc[rowid + j * N_loc];
                    sendoffset++;
                }
                MPI_Send(sendbuf, send_count[idsend] * M0, MPI_DOUBLE , idsend, mpitag, MPI_COMM_WORLD);
                idsend++;
            }
        }
        while (recv_count[idsend] == 0 && idsend < np)
            idsend++;
        while (send_count[idrecv] == 0 && idrecv < np)
            idrecv++;
    }

    free(sendbuf);

    //内点相乘
    char char_N = 'N', char_T = 'T', char_V = 'V', char_U = 'U';
    double one, zero;
    one = 1; 
    zero = 0; 
    char matdescra[6] = {'G', 'U', 'U', 'F'}; //矩阵A的类型
    mkl_dcsrmm (&char_N, &N_loc, &M0, &N_loc, &one, matdescra, a_in, ja_in, ia_in, ia_in + 1, Y_loc, &N_loc, &zero, result, &N_loc);//work = A * Y;

    //外点相乘
    for (i = 0; i < nnz_out; i++)
    {
        rowid = ia_out[i];
        colid = ja_out[i];
        for (j = 0; j < M0; j++)
        {
            result[rowid + j * N_loc] =  result[rowid + j * N_loc] + a_out[i] * B_out[colid * M0 + j];
            //result[rowid + j * N_loc].i =  result[rowid + j * N_loc].i + a_out[i].i * B_out[colid * M0 + j].i;
        }
    }

    free(B_out);
}



/************************************
高斯点
*****************************************/
void Gauss(int nGauss, int i, double *xe, double *we)
{
    switch (nGauss)
    {

    case (3):
        switch (i)
        {
        case (0):
            *xe = 0;
            *we = 0.888888888888888888889;
            break;
        case (1):
            *xe = sqrt(0.6);
            *we = 0.555555555555555555555;
            break;
        case (2):
            *xe = -sqrt(0.6);
            *we = 0.5555555555555555555555;
            break;
        }
        break;


    case (4):
        switch (i)
        {
        case (0):
            *xe = -0.339981043584856264792;
            *we = 0.652145154862546142644;
            break;
        case (1):
            *xe = 0.339981043584856264792;
            *we = 0.652145154862546142644;
            break;
        case (2):
            *xe = -0.861136311594052575248;
            *we = 0.347854845137453857383;
            break;
        case (3):
            *xe = 0.861136311594052575248;
            *we = 0.347854845137453857383;
            break;
        }
        break;


    case (5):
        switch (i)
        {
        case (0):
            *xe = 0;
            *we = 0.568888888888888888889;
            break;
        case (1):
            *xe = 0.538469310105683091018;
            *we = 0.4786287;
            break;
        case (2):
            *xe = -0.538469310105683091018;
            *we = 0.47862878;
            break;
        case (3):
            *xe = 0.906179845938663992811;
            *we = 0.2369269;
            break;
        case (4):
            *xe = -0.906179845938663992811;
            *we = 0.2369269;
            break;
        }
        break;


    case (6):
        switch (i)
        {
        case (0):
            *xe = -0.661209386466264513688;
            *we = 0.360761573048138607569;
            break;
        case (1):
            *xe = 0.661209386466264513688;
            *we = 0.360761573048138607569;
            break;
        case (2):
            *xe = -0.238619186083196908630;
            *we = 0.467913934572691047389;
            break;
        case (3):
            *xe = 0.238619186083196908630;
            *we = 0.467913934572691047389;
            break;
        case (4):
            *xe = -0.932469514203152027832;
            *we = 0.171324492379170345043;
            break;
        case (5):
            *xe = 0.932469514203152027832;
            *we = 0.171324492379170345043;
            break;
        }
        break;


    case (8):
        switch (i)
        {
        case (0):
            *xe = -0.183434642495649804936;
            *we = 0.362683783378361982976;
            break;
        case (1):
            *xe = 0.183434642495649804936;
            *we = 0.362683783378361982976;
            break;
        case (2):
            *xe = -0.525532409916328985830;
            *we = 0.313706645877887287338;
            break;
        case (3):
            *xe = 0.525532409916328985830;
            *we = 0.313706645877887287338;
            break;
        case (4):
            *xe = -0.796666477413626739567;
            *we = 0.222381034453374470546;
            break;
        case (5):
            *xe = 0.796666477413626739567;
            *we = 0.222381034453374470546;
            break;
        case (6):
            *xe = -0.960289856497536231661;
            *we = 0.101228536290376259154;
            break;
        case (7):
            *xe = 0.960289856497536231661;
            *we = 0.101228536290376259154;
            break;
        }
        break;


    case (10):
        switch (i)
        {
        case (0):
            *xe = -0.148874338981631210881;
            *we = 0.295524224714752870187;
            break;
        case (1):
            *xe = 0.148874338981631210881;
            *we = 0.295524224714752870187;
            break;
        case (2):
            *xe = -0.433395394129247190794;
            *we = 0.269266719309996355105;
            break;
        case (3):
            *xe = 0.433395394129247190794;
            *we = 0.269266719309996355105;
            break;
        case (4):
            *xe = -0.6794095682990244062070;
            *we = 0.219086362515982044000;
            break;
        case (5):
            *xe = 0.6794095682990244062070;
            *we = 0.219086362515982044000;
            break;
        case (6):
            *xe = -0.865063366688984510759;
            *we = 0.149451349150580593150;
            break;
        case (7):
            *xe = 0.865063366688984510759;
            *we = 0.149451349150580593150;
            break;
        case (8):
            *xe = -0.973906528517171720066;
            *we = 0.0666713443086881375920;
            break;
        case (9):
            *xe = 0.973906528517171720066;
            *we = 0.0666713443086881375920;
            break;
        }
        break;


    case (12):
        switch (i)
        {
        case (0):
            *xe = -0.125233408511468915478;
            *we = 0.249147045813402785006;
            break;
        case (1):
            *xe = 0.125233408511468915478;
            *we = 0.249147045813402785006;
            break;
        case (2):
            *xe = -0.367831498998180193757;
            *we = 0.233492536538354808758;
            break;
        case (3):
            *xe = 0.367831498998180193757;
            *we = 0.233492536538354808758;
            break;
        case (4):
            *xe = -0.587317954286617447312;
            *we = 0.203167426723065921743;
            break;
        case (5):
            *xe = 0.587317954286617447312;
            *we = 0.203167426723065921743;
            break;
        case (6):
            *xe = -0.769902674194304687059;
            *we = 0.160078328543346226338;
            break;
        case (7):
            *xe = 0.769902674194304687059;
            *we = 0.160078328543346226338;
            break;
        case (8):
            *xe = -0.904117256370474856682;
            *we = 0.106939325995318430960;
            break;
        case (9):
            *xe = 0.904117256370474856682;
            *we = 0.106939325995318430960;
            break;
        case (10):
            *xe = -0.981560634246719250712;
            *we = 0.0471753363865118271952;
            break;
        case (11):
            *xe = 0.981560634246719250712;
            *we = 0.0471753363865118271952;
            break;
        }
        break;


    case (16):
        switch (i)
        {
        case (0):
            *xe = -0.0950125098376374401877;
            *we = 0.189450610455068496287;
            break;
        case (1):
            *xe = 0.0950125098376374401877;
            *we = 0.189450610455068496287;
            break;
        case (2):
            *xe = -0.281603550779258913231;
            *we = 0.182603415044923588872;
            break;
        case (3):
            *xe = 0.281603550779258913231;
            *we = 0.182603415044923588872;
            break;
        case (4):
            *xe = -0.458016777657227386350;
            *we = 0.169156519395002538183;
            break;
        case (5):
            *xe = 0.458016777657227386350;
            *we = 0.169156519395002538183;
            break;
        case (6):
            *xe = -0.617876244402643748452;
            *we = 0.149595988816576732080;
            break;
        case (7):
            *xe = 0.617876244402643748452;
            *we = 0.149595988816576732080;
            break;
        case (8):
            *xe = -0.755404408355003033891;
            *we = 0.124628971255533872056;
            break;
        case (9):
            *xe = 0.755404408355003033891;
            *we = 0.124628971255533872056;
            break;
        case (10):
            *xe = -0.865631202387831743866;
            *we = 0.0951585116824927848073;
            break;
        case (11):
            *xe = 0.865631202387831743866;
            *we = 0.0951585116824927848073;
            break;
        case (12):
            *xe = -0.944575023073232576090;
            *we = 0.0622535239386478928628;
            break;
        case (13):
            *xe = 0.944575023073232576090;
            *we = 0.0622535239386478928628;
            break;
        case (14):
            *xe = -0.989400934991649932601;
            *we = 0.0271524594117540948514;
            break;
        case (15):
            *xe = 0.989400934991649932601;
            *we = 0.0271524594117540948514;
            break;
        }
        break;


    case (20):
        switch (i)
        {
        case (0):
            *xe = -0.0765265211334973337513;
            *we = 0.152753387130725850699;
            break;
        case (1):
            *xe = 0.0765265211334973337513;
            *we = 0.152753387130725850699;
            break;
        case (2):
            *xe = -0.227785851141645078076;
            *we = 0.149172986472603746785;
            break;
        case (3):
            *xe = 0.227785851141645078076;
            *we = 0.149172986472603746785;
            break;
        case (4):
            *xe = -0.373706088715419560662;
            *we = 0.142096109318382051326;
            break;
        case (5):
            *xe = 0.373706088715419560662;
            *we = 0.142096109318382051326;
            break;
        case (6):
            *xe = -0.510867001950827097985;
            *we = 0.131688638449176626902;
            break;
        case (7):
            *xe = 0.510867001950827097985;
            *we = 0.131688638449176626902;
            break;
        case (8):
            *xe = -0.636053680726515025467;
            *we = 0.118194531961518417310;
            break;
        case (9):
            *xe = 0.636053680726515025467;
            *we = 0.118194531961518417310;
            break;
        case (10):
            *xe = -0.746331906460150792634;
            *we = 0.101930119817240435039;
            break;
        case (11):
            *xe = 0.746331906460150792634;
            *we = 0.101930119817240435039;
            break;
        case (12):
            *xe = -0.839116971822218823420;
            *we = 0.0832767415767047487264;
            break;
        case (13):
            *xe = 0.839116971822218823420;
            *we = 0.0832767415767047487264;
            break;
        case (14):
            *xe = -0.912234428251325905857;
            *we = 0.0626720483341090635663;
            break;
        case (15):
            *xe = 0.912234428251325905857;
            *we = 0.0626720483341090635663;
            break;
        case (16):
            *xe = -0.963971927277913791287;
            *we = 0.0406014298003869413320;
            break;
        case (17):
            *xe = 0.963971927277913791287;
            *we = 0.0406014298003869413320;
            break;
        case (18):
            *xe = -0.993128599185094924776;
            *we = 0.0176140071391521183115;
            break;
        case (19):
            *xe = 0.993128599185094924776;
            *we = 0.0176140071391521183115;
            break;
        }
        break;


    case (24):
        switch (i)
        {
        case (0):
            *xe = -0.0640568928626056260827;
            *we = 0.127938195346752156976;
            break;
        case (1):
            *xe = 0.0640568928626056260827;
            *we = 0.127938195346752156976;
            break;
        case (2):
            *xe = -0.191118867473616309153;
            *we = 0.125837456346828296117;
            break;
        case (3):
            *xe = 0.191118867473616309153;
            *we = 0.125837456346828296117;
            break;
        case (4):
            *xe = -0.315042679696163374398;
            *we = 0.121670472927803391202;
            break;
        case (5):
            *xe = 0.315042679696163374398;
            *we = 0.121670472927803391202;
            break;
        case (6):
            *xe = -0.433793507626045138478;
            *we = 0.115505668053725601353;
            break;
        case (7):
            *xe = 0.433793507626045138478;
            *we = 0.115505668053725601353;
            break;
        case (8):
            *xe = -0.545421471388839535649;
            *we = 0.107444270115965634785;
            break;
        case (9):
            *xe = 0.545421471388839535649;
            *we = 0.107444270115965634785;
            break;
        case (10):
            *xe = -0.648093651936975569268;
            *we = 0.0976186521041138882720;
            break;
        case (11):
            *xe = 0.648093651936975569268;
            *we = 0.0976186521041138882720;
            break;
        case (12):
            *xe = -0.740124191578554364260;
            *we = 0.0861901615319532759152;
            break;
        case (13):
            *xe = 0.740124191578554364260;
            *we = 0.0861901615319532759152;
            break;
        case (14):
            *xe = -0.820001985973902921981;
            *we = 0.0733464814110803057346;
            break;
        case (15):
            *xe = 0.820001985973902921981;
            *we = 0.0733464814110803057346;
            break;
        case (16):
            *xe = -0.886415527004401034190;
            *we = 0.0592985849154367807461;
            break;
        case (17):
            *xe = 0.886415527004401034190;
            *we = 0.0592985849154367807461;
            break;
        case (18):
            *xe = -0.938274552002732758539;
            *we = 0.0442774388174198061695;
            break;
        case (19):
            *xe = 0.938274552002732758539;
            *we = 0.0442774388174198061695;
            break;
        case (20):
            *xe = -0.974728555971309498199;
            *we = 0.0285313886289336631809;
            break;
        case (21):
            *xe = 0.974728555971309498199;
            *we = 0.0285313886289336631809;
            break;
        case (22):
            *xe = -0.995187219997021360195;
            *we = 0.0123412297999871995469;
            break;
        case (23):
            *xe = 0.995187219997021360195;
            *we = 0.0123412297999871995469;
            break;
        }
        break;


    case (32):
        switch (i)
        {
        case (0):
            *xe = -0.0640568928626056260827;
            *we = 0.0965400885147278005666;
            break;
        case (1):
            *xe = 0.0640568928626056260827;
            *we = 0.0965400885147278005666;
            break;
        case (2):
            *xe = -0.144471961582796493484;
            *we = 0.0956387200792748594185;
            break;
        case (3):
            *xe = 0.144471961582796493484;
            *we = 0.0956387200792748594185;
            break;
        case (4):
            *xe = -0.239287362252137074544;
            *we = 0.0938443990808045656367;
            break;
        case (5):
            *xe = 0.239287362252137074544;
            *we = 0.0938443990808045656367;
            break;
        case (6):
            *xe = -0.331868602282127649782;
            *we = 0.0911738786957638847129;
            break;
        case (7):
            *xe = 0.331868602282127649782;
            *we = 0.0911738786957638847129;
            break;
        case (8):
            *xe = -0.421351276130635345353;
            *we = 0.0876520930044038111450;
            break;
        case (9):
            *xe = 0.421351276130635345353;
            *we = 0.0876520930044038111450;
            break;
        case (10):
            *xe = -0.506899908932229390044;
            *we = 0.0833119242269467552223;
            break;
        case (11):
            *xe = 0.506899908932229390044;
            *we = 0.0833119242269467552223;
            break;
        case (12):
            *xe = -0.587715757240762329066;
            *we = 0.0781938957870703064685;
            break;
        case (13):
            *xe = 0.587715757240762329066;
            *we = 0.0781938957870703064685;
            break;
        case (14):
            *xe = -0.663044266930215200960;
            *we = 0.0723457941088485062287;
            break;
        case (15):
            *xe = 0.663044266930215200960;
            *we = 0.0723464814110803057346;
            break;
        case (16):
            *xe = -0.732182118740289680412;
            *we = 0.0658222227763618468406;
            break;
        case (17):
            *xe = 0.732182118740289680412;
            *we = 0.0658222227763618468406;
            break;
        case (18):
            *xe = -0.794483795967942406965;
            *we = 0.0586840934785355471448;
            break;
        case (19):
            *xe = 0.794483795967942406965;
            *we = 0.0586840934785355471448;
            break;
        case (20):
            *xe = -0.849367613732569970160;
            *we = 0.0509980592623761761959;
            break;
        case (21):
            *xe = 0.849367613732569970160;
            *we = 0.0509980592623761761959;
            break;
        case (22):
            *xe = -0.896321155766052123971;
            *we = 0.0428358980222266806557;
            break;
        case (23):
            *xe = 0.896321155766052123971;
            *we = 0.0428358980222266806557;
            break;
        case (24):
            *xe = -0.934906075937739689159;
            *we = 0.0342738629130214331033;
            break;
        case (25):
            *xe = 0.934906075937739689159;
            *we = 0.0342738629130214331033;
            break;
        case (26):
            *xe = -0.964762255587506430761;
            *we = 0.0253920653092620594561;
            break;
        case (27):
            *xe = 0.964762255587506430761;
            *we = 0.0253920653092620594561;
            break;
        case (28):
            *xe = -0.985611511545268335400;
            *we = 0.0162743947309056706058;
            break;
        case (29):
            *xe = 0.985611511545268335400;
            *we = 0.0162743947309056706058;
            break;
        case (30):
            *xe = -0.997263861849481563534;
            *we = 0.00701861000947009660028;
            break;
        case (31):
            *xe = 0.997263861849481563534;
            *we = 0.00701861000947009660028;
            break;
        }
        break;


    case (48):
        switch (i)
        {
        case (0):
            *xe = -0.0323801709628693620343;
            *we = 0.0647376968126839225006;
            break;
        case (1):
            *xe = 0.0323801709628693620343;
            *we = 0.0647376968126839225006;
            break;
        case (2):
            *xe = -0.0970046992094626989322;
            *we = 0.0644661644359500822082;
            break;
        case (3):
            *xe = 0.0970046992094626989322;
            *we = 0.0644661644359500822082;
            break;
        case (4):
            *xe = -0.161222356068891718055;
            *we = 0.0639242385846481866207;
            break;
        case (5):
            *xe = 0.161222356068891718055;
            *we = 0.0639242385846481866207;
            break;
        case (6):
            *xe = -0.224763790394689061224;
            *we = 0.0631141922862540256548;
            break;
        case (7):
            *xe = 0.224763790394689061224;
            *we = 0.0631141922862540256548;
            break;
        case (8):
            *xe = -0.287362487355455576728;
            *we = 0.0620394231598926639029;
            break;
        case (9):
            *xe = 0.287362487355455576728;
            *we = 0.0620394231598926639029;
            break;
        case (10):
            *xe = -0.348755886292160738148;
            *we = 0.0607044391658938800517;
            break;
        case (11):
            *xe = 0.348755886292160738148;
            *we = 0.0607044391658938800517;
            break;
        case (12):
            *xe = -0.408686481990716729925;
            *we = 0.0591148396983956357477;
            break;
        case (13):
            *xe = 0.408686481990716729925;
            *we = 0.0591148396983956357477;
            break;
        case (14):
            *xe = -0.466902904750958404535;
            *we = 0.0572772921004032157044;
            break;
        case (15):
            *xe = 0.466902904750958404535;
            *we = 0.0572772921004032157044;
            break;
        case (16):
            *xe = -0.523160974722233033658;
            *we = 0.0551995036999841628676;
            break;
        case (17):
            *xe = 0.523160974722233033658;
            *we = 0.0551995036999841628676;
            break;
        case (18):
            *xe = -0.577224726083972703838;
            *we = 0.0528901894851936670964;
            break;
        case (19):
            *xe = 0.577224726083972703838;
            *we = 0.0528901894851936670964;
            break;
        case (20):
            *xe = -0.628867396776513624013;
            *we = 0.0503590355538544749590;
            break;
        case (21):
            *xe = 0.628867396776513624013;
            *we = 0.0503590355538544749590;
            break;
        case (22):
            *xe = -0.677872379632663905208;
            *we = 0.0476166584924904748267;
            break;
        case (23):
            *xe = 0.677872379632663905208;
            *we = 0.0476166584924904748267;
            break;
        case (24):
            *xe = -0.724034130923814654658;
            *we = 0.0446745608566942804201;
            break;
        case (25):
            *xe = 0.724034130923814654658;
            *we = 0.0446745608566942804201;;
            break;
        case (26):
            *xe = -0.767159032515740339276;
            *we = 0.0415450829434647492133;
            break;
        case (27):
            *xe = 0.767159032515740339276;
            *we = 0.0415450829434647492133;
            break;
        case (28):
            *xe = -0.807066204029442627087;
            *we = 0.0382413510658307063158;
            break;
        case (29):
            *xe = 0.807066204029442627087;
            *we = 0.0382413510658307063158;
            break;
        case (30):
            *xe = -0.843588261624393530704;
            *we = 0.0347772225647704388909;
            break;
        case (31):
            *xe = 0.843588261624393530704;
            *we = 0.0347772225647704388909;
            break;
        case (32):
            *xe = -0.876572020274247885885;
            *we = 0.0311672278327980889025;
            break;
        case (33):
            *xe = 0.876572020274247885885;
            *we = 0.0311672278327980889025;
            break;
        case (34):
            *xe = -0.905879136715569672805;
            *we = 0.0274265097083569482001;
            break;
        case (35):
            *xe = 0.905879136715569672805;
            *we = 0.0274265097083569482001;
            break;
        case (36):
            *xe = -0.931386690706554333107;
            *we = 0.0235707608393243791410;
            break;
        case (37):
            *xe = 0.931386690706554333107;
            *we = 0.0235707608393243791410;
            break;
        case (38):
            *xe = -0.952987703160430860724;
            *we = 0.0196161604573555278142;
            break;
        case (39):
            *xe = 0.952987703160430860724;
            *we = 0.0196161604573555278142;
            break;
        case (40):
            *xe = -0.970591592546247250472;
            *we = 0.0155793157229438487279;
            break;
        case (41):
            *xe = 0.970591592546247250472;
            *we = 0.0155793157229438487279;
            break;
        case (42):
            *xe = -0.984124583722826857765;
            *we = 0.0114772345792345394895;
            break;
        case (43):
            *xe = 0.984124583722826857765;
            *we = 0.0114772345792345394895;
            break;
        case (44):
            *xe = -0.993530172266350757526;
            *we = 0.00732755390127626210220;
            break;
        case (45):
            *xe = 0.993530172266350757526;
            *we = 0.00732755390127626210220;
            break;
        case (46):
            *xe = -0.998771007252426118580;
            *we = 0.00315334605230583863260;
            break;
        case (47):
            *xe = 0.998771007252426118580;
            *we = 0.00315334605230583863260;
            break;
        }
        break;
    }

}



