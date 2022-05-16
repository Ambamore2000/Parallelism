/* File: mpi_vector_ops.c
 * COMP 137-1 Spring 2020
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
char* infile = NULL;
char* outfile = NULL;

int readInputFile(char* filename, long* n_p, double* x_p, double** A_p, double** B_p)
{
    long i;
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) return 0;
    fscanf(fp, "%ld\n", n_p);
    fscanf(fp, "%lf\n", x_p);
    *A_p = malloc(*n_p*sizeof(double));
    *B_p = malloc(*n_p*sizeof(double));
    for (i=0; i<*n_p; i++) fscanf(fp, "%lf\n", (*A_p)+i);
    for (i=0; i<*n_p; i++) fscanf(fp, "%lf\n", (*B_p)+i);
    fclose(fp);
    return 1;
}

int writeOutputFile(char* filename, long n, double y, double* C, double* D)
{
    long i;
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) return 0;
    fprintf(fp, "%ld\n", n);
    fprintf(fp, "%lf\n", y);
    for (i=0; i<n; i++) fprintf(fp, "%lf\n", C[i]);
    for (i=0; i<n; i++) fprintf(fp, "%lf\n", D[i]);
    fclose(fp);
    return 1;
}

void serialSolution(long n, double x, double* A, double* B,
                    double* y, double* C, double* D)
{
    long i;
    double dp = 0.0;
    for (i=0; i<n; i++)
    {
        C[i] = x*A[i];
        D[i] = x*B[i];
        dp += A[i]*B[i];
    }
    *y = dp;
}

int main(int argc, char* argv[])
{
    int my_rank, comm_sz;
    double  local_y;
    double* local_A;
    double* local_B;
    double* local_C;
    double* local_D;
    long    n=0;     /* size of input arrays */
    double  x;       /* input scalar */
    double* A;       /* input vector */
    double* B;       /* input vector */
    double* C;       /* output vector xA */
    double* D;       /* output vector xB */
    double  y;       /* output scalar A dot B */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0) {
        /* read input data */
        if (argc<3)
        {
            n = -1;
            fprintf(stderr, "Command line arguments are required.\n");
            fprintf(stderr, "argv[1] = name of input file\n");
            fprintf(stderr, "argv[2] = name of input file\n");
        }
        else
        {
            infile = argv[1];
            outfile = argv[2];
            if (!readInputFile(infile, &n, &x, &A, &B))
            {
                fprintf(stderr, "Error opening input files. Aborting.\n");
                n = -1;
            }
        }
        
        if (comm_sz != 2 && comm_sz != 4 && comm_sz != 8 && comm_sz != 16
        && comm_sz != 32 && comm_sz != 64 && comm_sz != 128) {
            fprintf(stderr, "Invalid p (comm_sz). Aborting.\n");
            n = -1;
        }
    }
    
    MPI_Bcast(&x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    if (n < 0)
    {
        fprintf(stderr, "Aborting task due to input errors.\n");
        exit(1);
    }
    
    local_A = malloc((n/comm_sz)*sizeof(double));
    local_B = malloc((n/comm_sz)*sizeof(double));
    
    MPI_Scatter(A, (n/comm_sz), MPI_DOUBLE, local_A, (n/comm_sz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, (n/comm_sz), MPI_DOUBLE, local_B, (n/comm_sz), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    local_C = malloc((n/comm_sz)*sizeof(double));
    local_D = malloc((n/comm_sz)*sizeof(double));

    serialSolution((n/comm_sz), x, local_A, local_B, &local_y, local_C, local_D);
    
    MPI_Reduce(&local_y, &y, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    C = malloc(n*sizeof(double));
    D = malloc(n*sizeof(double));
    
    MPI_Gather(local_C, (n/comm_sz), MPI_DOUBLE, C, (n/comm_sz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(local_D, (n/comm_sz), MPI_DOUBLE, D, (n/comm_sz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Finalize();
    
    if (my_rank == 0) {
        if (!writeOutputFile(outfile, n, y, C, D)) {
            fprintf(stderr, "Error opening output file. Aborting.\n");
            exit(1);
        }
    }
    
    /* free all dynamic memory allocation */
    free(A);
    free(B);
    free(C);
    free(D);
    free(local_A);
    free(local_B);
    free(local_C);
    free(local_D);
    return 0;
}
