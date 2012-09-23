/* 
*  Program to perform three different numerical matrix operations in two ways each.
   One naive way, and a second smart one with tiling, to improve cache locality and 
   hence, execution time.
*/
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>


/* function prototypes */

float** read_matrix(float **array, int *arr_size, const char *filename);
void write_matrix(float **array, int arr_size, const char *filename);

float ** naive_matmul(float **arr1, float **arr2, float **arr3, int arr_size);
float ** tiled_matmul(float **arr1, float **arr2, float **arr3, int arr_size, int tile_size);

void naive_transpose(float **array, int arr_size);
void tiled_transpose(float **array, int arr_size, int tile_size);
void transpose(float **array, int index_x, int index_y, int subarr_size);
void exchange (float **array, int index_x, int index_y, int index_x2, int index_y2, int tile_size);

void naive_stencil(float **array, int arr_size);
void tiled_stencil(float **array, int arr_size, int tile_size);

/* main function */

int main(int argc, char *argv[]){

    int tilesize, i,j, len ;
    float **arr_A, **arr_B, **arr_C, **arr_A1, **arr_E, **arr_D1;
	int arrA_size, arrB_size, arrC_size, arrE_size, arrA1_size;
	struct timeval tim;
	
	if (argc < 2) {
       fprintf(stderr,"Name of file or tile size not provided.\n", argv[0]);
       exit(0);
    	}
    	
    char *filename;	
    filename = (char*) malloc (20);
    strcpy (filename, argv[1]);	
    
    tilesize = atoi(argv[2]);
    
    strcat (filename, "_A.txt"); 
    
    arr_A = read_matrix(arr_A, &arrA_size, filename);			//Read array A from file
	
	/* Naive Transpose */
	gettimeofday(&tim, NULL); 
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0); 
	naive_transpose (arr_A, arrA_size);				
	gettimeofday(&tim, NULL);  
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);    
	printf("%.6lf Naive Transpose - seconds elapsed\n", t2-t1);
	double t3 = t2-t1;
	
    write_matrix(arr_A, arrA_size, "out_A.txt");   
  
    arr_A = read_matrix(arr_A, &arrA_size, filename);			//Read array A from file
    
    /* do tiled matrix transpose here */
    gettimeofday(&tim, NULL); 
    t1=tim.tv_sec+(tim.tv_usec/1000000.0); 
    tiled_transpose(arr_A, arrA_size, tilesize);
    gettimeofday(&tim, NULL);  
    t2=tim.tv_sec+(tim.tv_usec/1000000.0);     
	printf("%.6lf Tiled Transpose - seconds elapsed\n", t2-t1);
	double t4 = t2-t1;
	
	printf("%.6lf Transpose - Speedup\n", t3/t4);

    write_matrix(arr_A, arrA_size, "out_A_t.txt");   

    /* Elementary Matrix Multiplication */
    
    arr_A1 = read_matrix(arr_A1, &arrA1_size, filename);	
    
    strcpy (filename, argv[1]);
    strcat (filename, "_B.txt");      
    
    arr_B = read_matrix(arr_B, &arrB_size, filename);			//Read array B and C from file
    
    
    if (arrB_size !=arrA1_size){
    	  	fprintf(stderr,"Matrix multiplication not possible with differently sized arrays");
       		exit(0);
    	}
    
    gettimeofday(&tim, NULL); 
    t1=tim.tv_sec+(tim.tv_usec/1000000.0); 	
    arr_C = naive_matmul(arr_A1, arr_B, arr_C, arrB_size);
    gettimeofday(&tim, NULL);  
    t2=tim.tv_sec+(tim.tv_usec/1000000.0);     
    printf("%.6lf Naive Transpose - seconds elapsed\n", t2-t1);
	t3 = t2-t1;
    
	write_matrix(arr_C, arrB_size, "out_C.txt");   

	/* do Tiled Matrix Multiplication here */
    	
    gettimeofday(&tim, NULL); 
    t1=tim.tv_sec+(tim.tv_usec/1000000.0);	
	arr_D1 = tiled_matmul(arr_A1, arr_B, arr_D1, arrB_size, tilesize);
	gettimeofday(&tim, NULL);  
    t2=tim.tv_sec+(tim.tv_usec/1000000.0);     
    printf("%.6lf Naive Transpose - seconds elapsed\n", t2-t1);
	t4 = t2-t1;
	
	printf("%.6lf Multiplication - Speedup\n", t3/t4);
	
	write_matrix(arr_D1, arrB_size, "out_C_t.txt");  
	
	/* 5-point naive stencil over 2D matrix*/
	
	strcpy (filename, argv[1]);
    strcat (filename, "_E.txt");
    arr_E = read_matrix(arr_E, &arrE_size, filename);
	
	naive_stencil(arr_E, arrE_size);
	
	write_matrix(arr_E, arrE_size, "out_E.txt");  
	
	/* 5-point tiled stencil over 2D matrix */
	
	arr_E = read_matrix(arr_E, &arrE_size, filename);
	
	tiled_stencil(arr_E, arrE_size, tilesize);
		
	write_matrix(arr_E, arrE_size, "out_E_t.txt");  
	
	
    /* Free allocated memory */
	    for (i = 0; i < arrA_size; i++){  		
	 		  free(arr_A[i]);  
  			  }  
    	free(arr_A);  
    	for (i = 0; i < arrB_size; i++){  		
	 		  free(arr_B[i]);  
  			  }  
    	free(arr_B);
    	for (i = 0; i < arrC_size; i++){  		
	 		  free(arr_C[i]);  
  			  }  
    	free(arr_C);  
				
	return 0;
}

/* Function to read a matrix from a text file
 * Takes filename as input
 * Populates the passed in array and returns it, also populates arr_size which is passed by reference.
*/

float** read_matrix(float **array, int *arr_size, const char *filename){
	
	int i, j;
	FILE *file = fopen (filename, "r" );						//Open a file for reading

	if (file!=NULL)
		{
			fscanf(file, "%d", arr_size);						//Read size of array from file
			array = (float**)malloc((*arr_size)*sizeof(float*));
			for (i= 0; i< (*arr_size); i++)
				array[i]= (float*)malloc((*arr_size)*sizeof(float));	//allocate memory for array
				
			while(!feof(file))
			{
				for (i=0;i<(*arr_size);i++)
					for (j=0;j<(*arr_size);j++)
					{
						fscanf(file, "%f", &array[i][j]);
					}
			}
		}
	else
		{
	   		fprintf(stderr,"File not found! -  %s \n",filename);
       		exit(0);
	    }
    
	return array;
}

/* Function to write a matrix to a text file.
 * Takes matrix, its size and filename as input.
*/

void write_matrix(float **array, int arr_size, const char *filename){
		
	int i, j;
	FILE *file_w = fopen (filename, "w" );						//Open a file for writing

	fprintf(file_w, "%d", arr_size);						//write size of array into file
	fprintf(file_w, "%s", "\n");

	for (i=0; i<arr_size; i++)
			for (j=0; j<arr_size; j++)				
				{				
					fprintf(file_w, "%.2f", array[i][j]);						
					fprintf(file_w, "\n");
				}
}

/* 
*  Function to calculate transpose of a given matrix without using tiles.
*  Takes matrix and its size as input.
*/

void naive_transpose(float **array, int arr_size){
	
	int i, j, temp;
	for (i=0; i<arr_size; i++) {
		for (j=i+1; j<arr_size; j++) {
			temp=array[j][i];
			array[j][i]=array[i][j];
			array[i][j] = temp;
			}
		}
}

/* Function to multiply 2 matrices the naive way.
 * Takes 2 matrices as input along with their size and the tile size as specified by the user.
 * Returns product matrix.
*/

float ** naive_matmul(float **arr1, float **arr2, float **arr3, int arr_size){

	int i, j, k;
	
	arr3 = (float**)malloc(arr_size*sizeof(float*));
	for (i= 0; i< arr_size; i++)
		arr3[i]= (float*)malloc(arr_size*sizeof(float));	//allocate memory for array
				
	for (i=0; i<arr_size; i++){
		for (j=0; j<arr_size; j++){
			for (k=0; k<arr_size; k++){ 
					arr3[i][j] = arr1[i][k]*arr2[k][j] + arr3[i][j];
					}
				}
			}
	
	return arr3;
}

/* Function to multiply 2 matrices using cache efficient tile algorithm.
 * Takes 2 matrices as input along with their size and the tile size as specified by the user.
 * Returns product matrix.
*/

float ** tiled_matmul(float **arr1, float **arr2, float **arr3, int arr_size, int tile_size){

	int i, j, k, i1, j1, k1;
	
	arr3 = (float**)malloc(arr_size*sizeof(float*));
	for (i= 0; i< arr_size; i++)
		arr3[i]= (float*)malloc(arr_size*sizeof(float));	//allocate memory for resultant array
				
	for (i=0; i<arr_size; i+=tile_size){
		for (j=0; j<arr_size; j+=tile_size){
			for (k=0; k<arr_size; k+=tile_size){
				for (i1=i; i1<i+tile_size; i1++){
					for (j1=j; j1<j+tile_size; j1++){
						for (k1=k; k1<k+tile_size; k1++){
							arr3[i1][j1] = arr1[i1][k1]*arr2[k1][j1] + arr3[i1][j1];
						}
					}
				}
			}
		}
	}
	
	return arr3;

}	

/* Function to transpose an array using cache efficient tile algorithm.
 * Takes matrix as input along with its size and the tile size as specified by the user.
*/

void tiled_transpose(float **array, int arr_size, int tile_size){
	
	int i, j, p, q, temp;
	for (i=0; i<arr_size; i+=tile_size) {
		
	 //for diagonals		
	for (p=i; p<i+tile_size; p++) {
		for (q=p+1; q<i+tile_size; q++) {
				temp=array[q][p];
				array[q][p]=array[p][q];
				array[p][q] = temp;
				}
		}
	
	for (j=i+tile_size; j< arr_size; j+=tile_size) {
	
		 transpose (array, j, i, tile_size);
 		 transpose (array, i, j, tile_size);
 		 exchange (array, j, i , i, j, tile_size);

			}
		}	
}

/* Helper function to calculate transpose of a sub array inside the matrix.
 * Takes the matrix, and the index of the starting point of the sub array as input.
*/

void transpose(float **array, int index_x, int index_y, int tile_size){
	
	int p, q, temp;
	
	 for (p=index_x+1; p<index_x+tile_size; p++) {
			for (q=index_y+1; q<index_y+tile_size; q++) {
		//		cout<<"Exchanging arr("<<index_x<<","<<q<<"and arr("<<p<<","<<index_y<<")"<<endl;
				temp=array[index_x][q];
				array[index_x][q]=array[p][index_y];
				array[p][index_y] = temp;
				}
			}
			
}

/* Helper function to exchange 2 sub arrays inside the matrix.
 * Takes the matrix, and the indices of the starting points of the 2 sub arrays as input.
*/

void exchange (float **array, int index_x, int index_y, int index_x2, int index_y2, int tile_size){

	int i, j, k, m, temp;

	for (i=index_x, k= index_x2; i< index_x + tile_size; i++, k++) {
		for (j=index_y, m= index_y2; j<index_y + tile_size; j++, m++){
			temp = array[i][j];
			array[i][j] = array[k][m];
			array[k][m] = temp;
		}
	}
	
}			

/* Function to perform the 5-point stencil on a matrix the naive way.
 * Takes the matrix and its size as input
*/

void naive_stencil(float **array, int arr_size){

	int i, j;
	
	for(i=1;i<arr_size-1;i++) {
			for(j=1;j<arr_size-1;j++) {
				array[i][j] = (array[i][j-1]+  array[i][j+1]+ array[i-1][j]+ array[i+1][j]+ array[i][j])/5;
			}
		}

}

/* Function to perform the 5-point stencil on a matrix using cache efficient tile algorithm.
 * Takes the matrix and its size as input.
*/

void tiled_stencil(float **array, int arr_size, int tile_size){

	int i, j, p, q;
	
	for (i=1; i<arr_size-1; i+=tile_size) {
		for (j=1; j<arr_size-1; j+=tile_size) {
			
			for(p=i;p<i+tile_size;p++) {
				for(q=j;q<j+tile_size;q++) {
					array[p][q] = (array[p][q-1] + array[p][q+1] + array[p-1][q] + array[p+1][q] + array[p][q])/5;
				}
			}			
	
		}
	}

}

		