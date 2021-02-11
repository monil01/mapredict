#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
//#define NUM_THREAD 4
#define false 0
#define true 1

#define NVL 1
#define VHEAP 2
#if MEM == NVL
#include <nvl.h>
#define NVL_PREFIX nvl
#elif MEM == VHEAP
#include <nvl-vheap.h>
#define NVL_PREFIX  
#else
#define NVL_PREFIX  
#endif

#ifndef NVLFILE
#define NVLFILE "_NVLFILEPATH_"
//#define NVLFILE "/opt/fio/scratch/f6l/bfs.nvl"
//#define NVLFILE "/opt/rd/scratch/f6l/bfs.nvl"
//#define NVLFILE "/tmp/f6l/bfs.nvl"
#endif

#ifndef NVLTMP
#define NVLTMP "/opt/fio/scratch/f6l"
#endif

#define DEBUG

#ifndef VERIFICATION
#define VERIFICATION 1
#endif

//macro for input graph4096.txt
//#define NUM_OF_NODES 4096
//#define EDGELIST_SIZE 24576
//macro for input graph1M.txt
//#define NUM_OF_NODES 1000000
//#define EDGELIST_SIZE 5999970
//macro for input graph4M.txt
//#define NUM_OF_NODES 4194304
//#define EDGELIST_SIZE 25159848
//macro for input graph16M.txt
//#define NUM_OF_NODES 16777216
//#define EDGELIST_SIZE 100666228
#ifndef NUM_OF_NODES
//#define NUM_OF_NODES 4096
#define NUM_OF_NODES 1000000
//#define NUM_OF_NODES 16777216
#endif
#ifndef EDGELIST_SIZE
//#define EDGELIST_SIZE 24576
#define EDGELIST_SIZE 5999970
//#define EDGELIST_SIZE 100666228
#endif

#ifndef HEAPSIZE
#define HEAPSIZE (NUM_OF_NODES*6*4*3 + EDGELIST_NODES*4*3)
#endif

#ifdef _OPENARC_

#if NUM_OF_NODES == 4096
	#pragma openarc #define NUM_OF_NODES 4096
#elif NUM_OF_NODES == 1000000
	#pragma openarc #define NUM_OF_NODES 1000000
#elif NUM_OF_NODES == 4194304
	#pragma openarc #define NUM_OF_NODES 4194304
#elif NUM_OF_NODES == 16777216
	#pragma openarc #define NUM_OF_NODES 16777216
#endif
#if EDGELIST_SIZE == 24576
	#pragma openarc #define EDGELIST_SIZE 24576
#elif EDGELIST_SIZE == 5999970
	#pragma openarc #define EDGELIST_SIZE 5999970
#elif EDGELIST_SIZE == 25159848
	#pragma openarc #define EDGELIST_SIZE 25159848
#elif EDGELIST_SIZE == 100666228
	#pragma openarc #define EDGELIST_SIZE 100666228
#endif

#endif

int no_of_nodes;
int edge_list_size;
FILE *fp;

#if MEM == NVL
struct root {
#if TXS
	int tid1;
	int tid2;
#endif
	nvl int * h_graph_nodes_starting;
	nvl int * h_graph_nodes_no_of_edges;
	nvl int * h_graph_mask;
	nvl int * h_updating_graph_mask;
	nvl int * h_graph_visited;
	nvl int * h_graph_edges;
	nvl int * h_cost;
};
#endif

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
	NVL_PREFIX int * h_graph_nodes_starting = 0;
	NVL_PREFIX int * h_graph_nodes_no_of_edges = 0;
	NVL_PREFIX int * h_graph_mask = 0;
	NVL_PREFIX int * h_updating_graph_mask = 0;
	NVL_PREFIX int * h_graph_visited = 0;
	NVL_PREFIX int * h_graph_edges = 0;
	NVL_PREFIX int * h_cost = 0;
#if MEM == NVL
	int * h_graph_nodes_starting_v;
	int * h_graph_nodes_no_of_edges_v;
	int * h_graph_mask_v;
	int * h_updating_graph_mask_v;
	int * h_graph_visited_v;
	int * h_graph_edges_v;
	int * h_cost_v;
#if TXS
	nvl int *tid1_nv = 0;
	nvl int *tid2_nv = 0;
#endif
#endif
	int start, edgeno;   
	int id,cost;
	int k;
	int stop;
	unsigned int i;
	int tid;
	FILE *fpo;
#ifdef DEBUG
	double start_time, end_time;
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
#if MEM == NVL
	nvl_heap_t * heap = nvl_create(NVLFILE, HEAPSIZE, 0600);
    if(!heap) {
        fprintf(stderr, "file %s already exists\n", NVLFILE);
		exit(1);
    }   
    nvl struct root *root_nv = 0;
    if( !(root_nv = nvl_alloc_nv(heap, 1, struct root)) 
        || !(h_graph_nodes_starting = nvl_alloc_nv(heap, no_of_nodes, int))
        || !(h_graph_nodes_no_of_edges = nvl_alloc_nv(heap, no_of_nodes, int))
        || !(h_graph_mask = nvl_alloc_nv(heap, no_of_nodes, int))
        || !(h_updating_graph_mask = nvl_alloc_nv(heap, no_of_nodes, int))
        || !(h_graph_visited = nvl_alloc_nv(heap, no_of_nodes, int)) )
    {   
        perror("nvl_alloc_nv failed");
		exit(1);
    }   
    nvl_set_root(heap, root_nv);
    root_nv->h_graph_nodes_starting = h_graph_nodes_starting;
    root_nv->h_graph_nodes_no_of_edges = h_graph_nodes_no_of_edges;
    root_nv->h_graph_mask = h_graph_mask;
    root_nv->h_updating_graph_mask = h_updating_graph_mask;
    root_nv->h_graph_visited = h_graph_visited;
#if TXS
	tid1_nv = &root_nv->tid1;
	tid2_nv = &root_nv->tid2;
#endif
	h_graph_nodes_starting_v = nvl_bare_hack(h_graph_nodes_starting);
	h_graph_nodes_no_of_edges_v = nvl_bare_hack(h_graph_nodes_no_of_edges);
	h_graph_mask_v = nvl_bare_hack(h_graph_mask);
	h_updating_graph_mask_v = nvl_bare_hack(h_updating_graph_mask);
	h_graph_visited_v = nvl_bare_hack(h_graph_visited);
#elif MEM == VHEAP
    nvl_vheap_t *vheap = nvl_vcreate(NVLTMP, HEAPSIZE);
    if(!vheap) {
        perror("nvl_vcreate failed");
        exit(1);   
    }   
    if( !(h_graph_nodes_starting = (int *)nvl_vmalloc(vheap, no_of_nodes*sizeof(int)))
    		|| !(h_graph_nodes_no_of_edges = (int *)nvl_vmalloc(vheap, no_of_nodes*sizeof(int)))
    		|| !(h_graph_mask = (int *)nvl_vmalloc(vheap, no_of_nodes*sizeof(int)))
    		|| !(h_updating_graph_mask = (int *)nvl_vmalloc(vheap, no_of_nodes*sizeof(int)))
    		|| !(h_graph_visited = (int *)nvl_vmalloc(vheap, no_of_nodes*sizeof(int))) )
    {   
        perror("nvl_vmalloc failed");
        exit(1);   
    }   
#else
	//h_graph_nodes = (struct Node*) malloc(sizeof(struct Node)*no_of_nodes);
	h_graph_nodes_starting = (int*) malloc(sizeof(int)*no_of_nodes);
	h_graph_nodes_no_of_edges = (int*) malloc(sizeof(int)*no_of_nodes);
	h_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	h_updating_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	h_graph_visited = (int*) malloc(sizeof(int)*no_of_nodes);
#endif

	// initalize the memory
	for( i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
#if MEM == NVL
		h_graph_nodes_starting_v[i] = start;
		h_graph_nodes_no_of_edges_v[i] = edgeno;
		h_graph_mask_v[i]=false;
		h_updating_graph_mask_v[i]=false;
		h_graph_visited_v[i]=false;
#else
		h_graph_nodes_starting[i] = start;
		h_graph_nodes_no_of_edges[i] = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
#endif
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);
	printf("edge_list_size = %d\n", edge_list_size);

#if MEM == NVL
    if( !(h_graph_edges = nvl_alloc_nv(heap, edge_list_size, int)) ) {
        perror("nvl_alloc_nv failed");
        exit(1);   
	}
    root_nv->h_graph_edges = h_graph_edges;
	h_graph_edges_v = nvl_bare_hack(h_graph_edges);
#elif MEM == VHEAP
	if( !(h_graph_edges = (int*) nvl_vmalloc(vheap, sizeof(int)*edge_list_size)) ) {
        perror("nvl_vmalloc failed");
        exit(1);   
	}
#else
	h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
#endif
	for(i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
#if MEM == NVL
		h_graph_edges_v[i] = id;
#else
		h_graph_edges[i] = id;
#endif
	}

	if(fp)
		fclose(fp);    


	// allocate mem for the result on host side
#if MEM == NVL
    if( !(h_cost = nvl_alloc_nv(heap, no_of_nodes, int)) ) {
        perror("nvl_alloc_nv failed");
        exit(1);   
	}
    root_nv->h_cost = h_cost;
	h_cost_v = nvl_bare_hack(h_cost);
#elif MEM == VHEAP
	if( !(h_cost = (int*) nvl_vmalloc(vheap, sizeof(int)*no_of_nodes)) ) {
        perror("nvl_vmalloc failed");
        exit(1);   
	}
#else
	h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
#endif
#if MEM == NVL
	for(i=0;i<no_of_nodes;i++)
		h_cost_v[i]=-1;
	h_cost_v[source]=0;
#else
	for(i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
#endif
	
	printf("Start traversing the tree\n");
	
#ifdef DEBUG
	start_time = gettime();
#endif
    
	k = 0;
#pragma acc data \
copyin(h_graph_visited[0:NUM_OF_NODES], h_graph_nodes_starting[0:NUM_OF_NODES], \
h_graph_nodes_no_of_edges[0:NUM_OF_NODES], h_graph_edges[0:EDGELIST_SIZE], \
h_graph_mask[0:NUM_OF_NODES], h_updating_graph_mask[0:NUM_OF_NODES]) \
copy(h_cost[0:NUM_OF_NODES]) 
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;

#if TXS
		assert(NUM_OF_NODES % ROWS_PER_TX == 0);
		for(tid = *tid1_nv; tid < NUM_OF_NODES; )
		{
#if (TXS == 1)
		#pragma nvl atomic heap(heap)
#else
		#pragma nvl atomic heap(heap) default(readonly) \
		backup(tid1_nv[0:1], h_cost[tid:ROWS_PER_TX], h_updating_graph_mask[tid:ROWS_PER_TX])
#endif
		for( int tid_sub=0; tid_sub<ROWS_PER_TX; ++tid_sub, ++tid, ++*tid1_nv )
		{
			if( *tid1_nv == (NUM_OF_NODES - 1) ) { *tid1_nv = -1; }
#else
		#pragma acc kernels loop independent, gang, worker, private(tid,i) \
		present(h_graph_visited[0:NUM_OF_NODES], h_graph_nodes_starting[0:NUM_OF_NODES], \
		h_graph_nodes_no_of_edges[0:NUM_OF_NODES], h_graph_edges[0:EDGELIST_SIZE], \
		h_graph_mask[0:NUM_OF_NODES], h_updating_graph_mask[0:NUM_OF_NODES], \
		h_cost[0:NUM_OF_NODES]) //async(0)
		for(tid = 0; tid < NUM_OF_NODES; tid++ )
		{
#endif
#if (MEM == NVL) && !POOR
			if (h_graph_mask_v[tid] == true){ 
				h_graph_mask_v[tid]=false;
				for(i=h_graph_nodes_starting_v[tid]; i<(h_graph_nodes_no_of_edges_v[tid] + h_graph_nodes_starting_v[tid]); i++)
				{
					int id = h_graph_edges_v[i];
					if(!h_graph_visited_v[id])
					{
#if TXS
						h_cost[id]=h_cost_v[tid]+1;
						h_updating_graph_mask[id]=true;
#else
						h_cost_v[id]=h_cost_v[tid]+1;
						h_updating_graph_mask_v[id]=true;
#endif
					}
				}
			}
#else
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
#endif
#if TXS
		}
#endif
		}
		//#pragma acc wait(0)
#if PERSIST && (MEM == NVL) && !POOR
		nvl_persist_hack(h_cost, no_of_nodes);
		nvl_persist_hack(h_updating_graph_mask, no_of_nodes);
#endif
		
#if TXS
		assert(NUM_OF_NODES % ROWS_PER_TX == 0);
		for(tid = *tid2_nv; tid < NUM_OF_NODES; )
		{
#if (TXS == 1)
		#pragma nvl atomic heap(heap)
#else
		#pragma nvl atomic heap(heap) default(readonly) \
		backup(tid2_nv[0:1], h_graph_mask[tid:ROWS_PER_TX], h_updating_graph_mask[tid:ROWS_PER_TX], h_graph_visited[tid:ROWS_PER_TX])
#endif
		for( int tid_sub=0; tid_sub<ROWS_PER_TX; ++tid_sub, ++tid, ++*tid2_nv )
		{
			if( *tid2_nv == (NUM_OF_NODES - 1) ) { *tid2_nv = -1; }
#else
		#pragma acc kernels loop independent, gang, worker, private(tid) \
		present(h_graph_visited[0:NUM_OF_NODES], \
		h_graph_mask[0:NUM_OF_NODES], h_updating_graph_mask[0:NUM_OF_NODES]) //async(0)
  		for(tid=0; tid< NUM_OF_NODES ; tid++ )
		{
#endif
#if (MEM == NVL) && !POOR
			if (h_updating_graph_mask_v[tid] == true){
#if TXS
				h_graph_mask[tid]=true;
				h_graph_visited[tid]=true;
				stop= stop | true;
				h_updating_graph_mask[tid]=false;
#else
				h_graph_mask_v[tid]=true;
				h_graph_visited_v[tid]=true;
				stop= stop | true;
				h_updating_graph_mask_v[tid]=false;
#endif
			}
#else
			if (h_updating_graph_mask[tid] == true){
				h_graph_mask[tid]=true;
				h_graph_visited[tid]=true;
				stop= stop | true;
				h_updating_graph_mask[tid]=false;
			}
#endif
#if TXS
		}
#endif
		}
		//#pragma acc wait(0)
#if PERSIST && (MEM == NVL) && !POOR
		nvl_persist_hack(h_graph_mask, no_of_nodes);
		nvl_persist_hack(h_graph_visited, no_of_nodes);
		nvl_persist_hack(h_updating_graph_mask, no_of_nodes);
#endif
		k++;
	}
	while(stop);


	#ifdef DEBUG
	end_time = gettime();
	printf("Kernel Executed %d times\n",k);
	printf("Accelerator Elapsed time = %lf sec\n", end_time - start_time);
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
			printf("Verification Failed\n");
		else
			printf("Verification Successful\n");

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
#if MEM == NVL
	nvl_close(heap);
#elif MEM == VHEAP
	nvl_vfree(vheap, h_graph_nodes_starting);
	nvl_vfree(vheap, h_graph_nodes_no_of_edges);
	nvl_vfree(vheap, h_graph_edges);
	nvl_vfree(vheap, h_graph_mask);
	nvl_vfree(vheap, h_updating_graph_mask);
	nvl_vfree(vheap, h_graph_visited);
	nvl_vfree(vheap, h_cost);
	nvl_vclose(vheap);
#else
	free( h_graph_nodes_starting);
	free( h_graph_nodes_no_of_edges);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
#endif
}
