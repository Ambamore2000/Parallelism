/* serial_vector_rotate.c
 * COMP 137 - OpenMPHW
 *
 * Program functionality:
 * Read 3D rotations and 3D vectors from a text file.
 * Build a rotation matrix and rotate all vectors using the rotation matrix.
 * Sum the rotated vectors.
 *
 * Program has a single command line argument, which is the
 * name of the data file that should be read and processed.
 *
 *  parallelize this program using pthreads.
 *
 * result for input1.txt:
 * Result = [-613.67, 28.55, -162.24]
 * result for input2.txt:
 * Result = [-661.51, -234.53, -179.22]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double start_time, run_time;
int thread_count = 0;
/* global variables */
char* input_file_name = NULL;
long num_vectors = 0;
float* original_vectors = NULL;
float* rotated_vectors = NULL;

/*--------------------------------------------------------------------*/

void processCommandLine(int argc, char* argv[]);
float* readInputDatafile(char* filename, long* num_vects, float angles[3]);
void multMatrixMatrix(float a[9], float b[9], float c[9]);
void multMatrixVector(float a[9], float b[3], float c[3]);
void addVectorVector(float a[3], float b[3], float c[3]);
void computeRotationMatrix(float angles[3], float rotation_matrix[9]);

/*--------------------------------------------------------------------*/

int main(int argc, char* argv[])
{


    /* allocate local variables */
    long v;
    float rotation_matrix[9];
    float angles[3];
    float result[3] = { 0.0f, 0.0f, 0.0f };
    float temp[3] = { 0.0f, 0.0f, 0.0f };

    /* check for command line argument */
	processCommandLine(argc, argv);

    /* read the file specified in the command line argument
       the reader function allocates the space for the input vectors */
    original_vectors = readInputDatafile(input_file_name, &num_vectors, angles);
    if (original_vectors == NULL)
    {
        fprintf(stderr, "could not read input file %s\n", input_file_name);
		exit(0);
    }


    /* allocated space for rotated vectors
       and compute the rotation transformation matrix */
    rotated_vectors = (float*)malloc(3*num_vectors*sizeof(float));
    computeRotationMatrix(angles, rotation_matrix);

	/* START OF CODE TO BE PARALLELIZED */
	start_time = omp_get_wtime();
#    pragma omp parallel for num_threads(thread_count)
    /* rotate all the vectors */
    for (v=0; v<num_vectors; v++) {
        multMatrixVector(rotation_matrix, &(original_vectors[v*3]), &(rotated_vectors[v*3]));
    }

#    pragma omp parallel for num_threads(thread_count) \
         reduction(+:result) private(temp)
    /* add all the vectors */
    for (v=0; v<num_vectors; v++)
    {
        addVectorVector(result, &(rotated_vectors[v*3]), temp);
        result[0] = temp[0]; result[1] = temp[1]; result[2] = temp[2];
    }
    run_time = omp_get_wtime() - start_time;
	/* END OF CODE TO BE PARALLELIZED */

    /* print results */
    printf("Result = [%0.2f, %0.2f, %0.2f]\n", result[0], result[1], result[2]);
    
    printf("Time = [%f]\n", run_time);
    

    /* clean up dynamic memory */
    free(original_vectors);
    free(rotated_vectors);


    return 0;
}

/*--------------------------------------------------------------------*/

/* print command line usage message and abort program. */
void usage(char* prog_name) {
	fprintf(stderr, "usage: %s <fn> <tc> \n", prog_name);
	fprintf(stderr, "   <fn> is name of the file containing the data to be processed\n");
	fprintf(stderr, "   <tc> is thread count\n");
	exit(0);
}

/* interpret command lines and store in shared variables */
void processCommandLine(int argc, char* argv[]) {
	if (argc != 3) usage(argv[0]);
	input_file_name = argv[1];
	thread_count = atoi(argv[2]);
}

/* read the input data file */
float* readInputDatafile(char* filename, long* num_vects, float angles[3])
{
	long i = 0, j = 0, n = 0;
	float* input_vectors;

	FILE* fp = fopen(filename, "r");
	if (fp == NULL) return NULL;
	fscanf(fp, "%f, %f, %f\n", &(angles[0]), &(angles[1]), &(angles[2]));
	fscanf(fp, "%ld\n", &n);
	*num_vects = n;
	input_vectors = (float*)malloc(3 * n * sizeof(float));
	for (i = 0; i<n; i++)
	{
		fscanf(fp, "%f, %f, %f\n", &(input_vectors[j]), &(input_vectors[j + 1]), &(input_vectors[j + 2]));
		j += 3;
	}
	fclose(fp);
	return input_vectors;
}

/*--------------------------------------------------------------------*/
/*
 * Matrix and vector mathematics
 * These functions are thread safe.
*/

void multMatrixMatrix(float a[9], float b[9], float c[9])
{
	/* c = a*b */
	c[0] = a[0] * b[0] + a[1] * b[3] + a[2] * b[6];
	c[1] = a[0] * b[1] + a[1] * b[4] + a[2] * b[7];
	c[2] = a[0] * b[2] + a[1] * b[5] + a[2] * b[8];
	c[3] = a[3] * b[0] + a[4] * b[3] + a[5] * b[6];
	c[4] = a[3] * b[1] + a[4] * b[4] + a[5] * b[7];
	c[5] = a[3] * b[2] + a[4] * b[5] + a[5] * b[8];
	c[6] = a[6] * b[0] + a[7] * b[3] + a[7] * b[6];
	c[7] = a[6] * b[1] + a[7] * b[4] + a[7] * b[7];
	c[8] = a[6] * b[2] + a[7] * b[5] + a[7] * b[8];
}

void multMatrixVector(float a[9], float b[3], float c[3])
{
	/* c = a*b */
	c[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	c[1] = a[3] * b[0] + a[4] * b[1] + a[5] * b[2];
	c[2] = a[6] * b[0] + a[7] * b[1] + a[8] * b[2];
}

void addVectorVector(float a[3], float b[3], float c[3])
{
	/* c = a + b */
	c[0] = a[0] + b[0];
	c[1] = a[1] + b[1];
	c[2] = a[2] + b[2];
}

void computeRotationMatrix(float angles[3], float rotation_matrix[9])
{
	float r = angles[2]; /* roll (radians) */
	float p = angles[0]; /* pitch (radians) */
	float y = angles[1]; /* yaw (radians) */
	float rx[9] =
	{ 1.0f,       0.0f,       0.0f,
		0.0f,       cosf(p),     -sinf(p),
		0.0f,       sinf(p),     cosf(p)
	};
	float ry[9] =
	{ cosf(y),    0.0f,       sinf(y),
		0.0f,       1.0f,       0.0f,
		-sinf(y),   0.0f,       cosf(y)
	};
	float rz[9] =
	{ cosf(r),    -sinf(r),   0.0f,
		sinf(r),    cosf(r),    0.0f,
		0.0f,       0.0f,       1.0f
	};
	float ry_rx[9];
	multMatrixMatrix(ry, rx, ry_rx);
	multMatrixMatrix(ry_rx, rz, rotation_matrix);
}
