ALL: out
myinc = -I/home/scjimq/petsc/include -I/home/scjimq/petsc/arch-linux2-c-debug/include -I/soft/mpi/openmpi/1.6.5/intel/include 
include /home/scjimq/petsc/lib/petsc/conf/variables
include /home/scjimq/petsc/lib/petsc/conf/rules

out :MAIN.o C_MainInit.o C_EIG.o petsc_slover.o  chkopts 
	mpicc  -o out MAIN.o C_MainInit.o  C_EIG.o petsc_slover.o   -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -openmp -liomp5 -limf -lsvml -lirc -lifcore -lpthread -lm  $(PETSC_LIB)
MAIN.o : MAIN.c
	mpicc -O3   -c MAIN.c
C_MainInit.o : C_MainInit.c
	mpicc -O3  -c C_MainInit.c
petsc_slover.o:petsc_slover.c  chkopts
	mpicc -O3    -c  petsc_slover.c $(myinc)
C_EIG.o : C_EIG.c   
	mpicc -O3   -c C_EIG.c 

clean::
	rm out *.o
