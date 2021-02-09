#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <sys/time.h>
#include "resilience.h"
#define false 0
#define true 1

#define DEBUG

#ifndef VERIFICATION
#define VERIFICATION 1
#endif
#ifndef R_MODE
#define R_MODE 0
#endif


#ifndef ENABLE_OPENACC
#define ENABLE_OPENACC 1
#endif

//macro for input graph4096.txt
//#define NUM_OF_NODES 4096
//#define EDGELIST_SIZE 24576
//macro for input graph1MW_6.txt
//#define NUM_OF_NODES 1000000
//#define EDGELIST_SIZE 5999970
#ifndef NUM_OF_NODES
#define NUM_OF_NODES 4096
#endif
#ifndef EDGELIST_SIZE
#define EDGELIST_SIZE 24576
#endif

#ifndef RES_REGION0
#define RES_REGION0 1
#endif
#ifndef RES_REGION1
#define RES_REGION1 0
#endif
#ifndef TOTAL_NUM_FAULTS
#define TOTAL_NUM_FAULTS    1
#endif
#ifndef NUM_FAULTYBITS
#define NUM_FAULTYBITS  1
#endif
#ifndef NUM_REPEATS
#define NUM_REPEATS 1
#endif
#ifndef _FTVAR
#define _FTVAR 0
#endif
#ifndef _FTKIND
#define _FTKIND 5
#endif
#ifndef _FTTHREAD
#define _FTTHREAD 0
#endif


#ifdef _OPENARC_

#if NUM_OF_NODES == 4096
	#pragma openarc #define NUM_OF_NODES 4096
#elif NUM_OF_NODES == 1000000
	#pragma openarc #define NUM_OF_NODES 1000000
#endif
#if EDGELIST_SIZE == 24576
	#pragma openarc #define EDGELIST_SIZE 24576
#elif EDGELIST_SIZE == 5999970
#pragma openarc #define EDGELIST_SIZE 5999970
#endif

#include "ftmacro.h"

#endif

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

void BFSGraph(int argc, char** argv);

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <num_threads> <input_file>\n", argv[0]);

}

double gettime() {
  struct timeval t;
  gettimeofday(&t,0);
  return t.tv_sec+t.tv_usec*1e-6;
}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	double start_time, end_time;
	start_time = gettime();
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
	end_time = gettime();
	printf("Total time = %lf sec \n", end_time - start_time);
	return 0;
}



////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
  char *input_f;
	int	 num_omp_threads;
	int source;
	//struct Node* h_graph_nodes;
	int * h_graph_nodes_starting;
	int * h_graph_nodes_no_of_edges;
	int *h_graph_mask;
	int *h_updating_graph_mask;
	int *h_graph_visited;
	int start, edgeno;   
	int id,cost;
	int* h_graph_edges;
	int k;
	int* h_cost;
	int stop;
	unsigned int i;
	int tid;
	FILE *fpo;
#ifdef DEBUG
	double start_time, end_time;
#endif
#if (R_MODE == 2) || (R_MODE == 3) 
    long count0 = 0;
    long count1 = 0;
#endif

	
	if(argc!=3){
	Usage(argc, argv);
	exit(0);
	}
    
	num_omp_threads = atoi(argv[1]);
	input_f = argv[2];
	
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	source = 0;

	fscanf(fp,"%d",&no_of_nodes);
	printf("no_of_nodes = %d\n", no_of_nodes);
   
	// allocate host memory
	//h_graph_nodes = (struct Node*) malloc(sizeof(struct Node)*no_of_nodes);
	h_graph_nodes_starting = (int*) malloc(sizeof(int)*no_of_nodes);
	h_graph_nodes_no_of_edges = (int*) malloc(sizeof(int)*no_of_nodes);
	h_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	h_updating_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	h_graph_visited = (int*) malloc(sizeof(int)*no_of_nodes);

	// initalize the memory
	for( i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes_starting[i] = start;
		h_graph_nodes_no_of_edges[i] = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);
	printf("edge_list_size = %d\n", edge_list_size);

	h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    


	// allocate mem for the result on host side
	h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
	
	printf("Start traversing the tree\n");
	
#ifdef DEBUG
	start_time = gettime();
#endif
    
	k = 0;

#if ENABLE_OPENACC == 1
#pragma acc data \
copyin(h_graph_visited[0:NUM_OF_NODES], h_graph_nodes_starting[0:NUM_OF_NODES], \
h_graph_nodes_no_of_edges[0:NUM_OF_NODES], h_graph_edges[0:EDGELIST_SIZE], \
h_graph_mask[0:NUM_OF_NODES], h_updating_graph_mask[0:NUM_OF_NODES]) \
copy(h_cost[0:NUM_OF_NODES]) 
#endif
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;

#if R_MODE == 0
		#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION0) ftdata(FTVAR0) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#elif R_MODE == 1
		#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION0) ftkind(FTKIND) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#elif R_MODE == 2
		#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION0) ftprofile(count0) ftdata(FTVAR0) num_faults(0) num_ftbits(0)
#elif R_MODE == 3
		#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION0) ftprofile(count0) ftkind(FTKIND) num_faults(0) num_ftbits(0)
#elif R_MODE == 4
		#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION0) ftpredict(FTCNT0) ftdata(FTVAR0) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#elif R_MODE == 5
		#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION0) ftpredict(FTCNT0) ftkind(FTKIND) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#endif

#if ENABLE_OPENACC == 1
		#pragma acc kernels loop independent, gang, worker, private(tid,i) \
		present(h_graph_visited[0:NUM_OF_NODES], h_graph_nodes_starting[0:NUM_OF_NODES], \
		h_graph_nodes_no_of_edges[0:NUM_OF_NODES], h_graph_edges[0:EDGELIST_SIZE], \
		h_graph_mask[0:NUM_OF_NODES], h_updating_graph_mask[0:NUM_OF_NODES], \
		h_cost[0:NUM_OF_NODES]) 
#endif
		for(tid = 0; tid < NUM_OF_NODES; tid++ )
		{
			if (h_graph_mask[tid] == true){ 
				h_graph_mask[tid]=false;
				for(i=h_graph_nodes_starting[tid]; i<(h_graph_nodes_no_of_edges[tid] + h_graph_nodes_starting[tid]); i++)
				{
					int id = h_graph_edges[i];
					if(!h_graph_visited[id])
					{
						h_cost[id]=h_cost[tid]+1;
						h_updating_graph_mask[id]=true;
					}
				}
			}
		}

#if R_MODE == 0
		#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION1) ftdata(FTVAR1) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#elif R_MODE == 1
		#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION1) ftkind(FTKIND) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#elif R_MODE == 2
		#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION1) ftprofile(count1) ftdata(FTVAR1) num_faults(0) num_ftbits(0)
#elif R_MODE == 3
		#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION1) ftprofile(count1) ftkind(FTKIND) num_faults(0) num_ftbits(0)
#elif R_MODE == 4
		#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION1) ftpredict(FTCNT1) ftdata(FTVAR1) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#elif R_MODE == 5
		#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION1) ftpredict(FTCNT1) ftkind(FTKIND) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#endif

#if ENABLE_OPENACC == 1
		#pragma acc kernels loop independent, gang, worker, private(tid) \
		present(h_graph_visited[0:NUM_OF_NODES], \
		h_graph_mask[0:NUM_OF_NODES], h_updating_graph_mask[0:NUM_OF_NODES])
#endif
  		for(tid=0; tid< NUM_OF_NODES ; tid++ )
		{
			if (h_updating_graph_mask[tid] == true){
				h_graph_mask[tid]=true;
				h_graph_visited[tid]=true;
				stop= stop | true;
				h_updating_graph_mask[tid]=false;
			}
		}
		k++;
	}
	while(stop);


#ifdef DEBUG
	end_time = gettime();
	printf("Kernel Executed %d times\n",k);
	printf("Accelerator Elapsed time = %lf sec\n", end_time - start_time);
#endif

#if (R_MODE == 2) || (R_MODE == 3) 
  printf("FT profile-count0 = %ld\n", count0);
  printf("FT profile-count1 = %ld\n", count1);
#endif


	if(VERIFICATION) {
		int * h_graph_nodes_starting_CPU;
		int * h_graph_nodes_no_of_edges_CPU;
		int *h_graph_mask_CPU;
		int *h_updating_graph_mask_CPU;
		int *h_graph_visited_CPU;
		int start, edgeno_CPU;
		int* h_graph_edges_CPU;
		int* h_cost_CPU;
		int good= 0;
	
		fp = fopen(input_f,"r");
		
		source = 0;

		fscanf(fp,"%d",&no_of_nodes);
		 
		// allocate host memory
		//h_graph_nodes = (struct Node*) malloc(sizeof(struct Node)*no_of_nodes);
		h_graph_nodes_starting_CPU = (int*) malloc(sizeof(int)*no_of_nodes);
		h_graph_nodes_no_of_edges_CPU = (int*) malloc(sizeof(int)*no_of_nodes);
		h_graph_mask_CPU = (int*) malloc(sizeof(int)*no_of_nodes);
		h_updating_graph_mask_CPU = (int*) malloc(sizeof(int)*no_of_nodes);
		h_graph_visited_CPU = (int*) malloc(sizeof(int)*no_of_nodes);

		// initalize the memory
		for( i = 0; i < no_of_nodes; i++) 
		{
			fscanf(fp,"%d %d",&start,&edgeno);
			h_graph_nodes_starting_CPU[i] = start;
			h_graph_nodes_no_of_edges_CPU[i] = edgeno;
			h_graph_mask_CPU[i]=false;
			h_updating_graph_mask_CPU[i]=false;
			h_graph_visited_CPU[i]=false;
		}

		//read the source node from the file
		fscanf(fp,"%d",&source);
		source=0;

		//set the source node as true in the mask
		h_graph_mask_CPU[source]=true;
		h_graph_visited_CPU[source]=true;

		fscanf(fp,"%d",&edge_list_size);

		h_graph_edges_CPU = (int*) malloc(sizeof(int)*edge_list_size);
		for(i=0; i < edge_list_size ; i++)
		{
			fscanf(fp,"%d",&id);
			fscanf(fp,"%d",&cost);
			h_graph_edges_CPU[i] = id;
		}

		if(fp)
			fclose(fp);    


		// allocate mem for the result on host side
		h_cost_CPU = (int*) malloc( sizeof(int)*no_of_nodes);
		for(i=0;i<no_of_nodes;i++)
			h_cost_CPU[i]=-1;
		h_cost_CPU[source]=0;
	
	
	#ifdef DEBUG
		start_time = gettime();
	#endif
		  
		k = 0;

		do
		{
			//if no thread changes this value then the loop stops
			stop=false;

			for(tid = 0; tid < NUM_OF_NODES; tid++ )
			{
				if (h_graph_mask_CPU[tid] == true){ 
					h_graph_mask_CPU[tid]=false;
					for(i=h_graph_nodes_starting_CPU[tid]; i<(h_graph_nodes_no_of_edges_CPU[tid] + h_graph_nodes_starting_CPU[tid]); i++)
					{
						int id = h_graph_edges_CPU[i];
						if(!h_graph_visited_CPU[id])
						{
							h_cost_CPU[id]=h_cost_CPU[tid]+1;
							h_updating_graph_mask_CPU[id]=true;
						}
					}
				}
			}


			for(tid=0; tid< NUM_OF_NODES ; tid++ )
			{
				if (h_updating_graph_mask_CPU[tid] == true){
					h_graph_mask_CPU[tid]=true;
					h_graph_visited_CPU[tid]=true;
					stop= stop | true;
					h_updating_graph_mask_CPU[tid]=false;
				}
			}
			k++;
		}
		while(stop);

		good=1;
		for(i=0; i<no_of_nodes; i++) {
			if(h_cost[i] != h_cost_CPU[i]) {
				good = 0;	
				break;;
			}
		}
		
		if(!good) 
			printf("Verification: Failed\n");
		else
			printf("Verification: Successful\n");

		free( h_graph_nodes_starting_CPU);
		free( h_graph_nodes_no_of_edges_CPU);
		free( h_graph_edges_CPU);
		free( h_graph_mask_CPU);
		free( h_updating_graph_mask_CPU);
		free( h_graph_visited_CPU);
		free( h_cost_CPU);

	}




	//Store the result into a file
	fpo = fopen("result.txt","w");
	for(i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");


	// cleanup memory
	free( h_graph_nodes_starting);
	free( h_graph_nodes_no_of_edges);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);

}

