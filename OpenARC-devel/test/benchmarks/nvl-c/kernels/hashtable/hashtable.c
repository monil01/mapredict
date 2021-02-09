#include <assert.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <sys/time.h>
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

#ifndef HASHFUNCTION
#define HASHFUNCTION 2
#endif

#ifndef NVLFILE
#define NVLFILE "_NVLFILEPATH_"
//#define NVLFILE "/opt/fio/scratch/f6l/hashtable.nvl"
//#define NVLFILE "/opt/rd/scratch/f6l/hashtable.nvl"
//#define NVLFILE "/tmp/f6l/hashtable.nvl"
#endif

#ifndef NVLTMP
#define NVLTMP "/opt/fio/scratch/f6l"
#endif

#ifndef HEAPSIZE
#define HEAPSIZE ((VALSIZE*NUMELE*32)+(NUMELE*256))
#endif

#define CHAR_BITS 8

double my_timer();
const char *my_itoa(int num);

size_t VALSIZE = 1;
size_t NUMELE = 1;
size_t nConflicts = 0;
size_t nDeletes = 0;

struct entry_s {
	NVL_PREFIX char *key;
	NVL_PREFIX int *value;
	NVL_PREFIX struct entry_s *next;
};

typedef struct entry_s entry_t;

struct hashtable_s {
	int size;
	NVL_PREFIX struct entry_s * NVL_PREFIX *table;	
};

typedef struct hashtable_s hashtable_t;

#if MEM == NVL
nvl_heap_t *heap = 0;
#elif MEM == VHEAP
nvl_vheap_t *vheap = 0;
#endif
#if (MEM == NVL) && !POOR
NVL_PREFIX hashtable_t *hashtable_nv = 0;
hashtable_t *hashtable = 0;
#else
NVL_PREFIX hashtable_t *hashtable = 0;
#endif

NVL_PREFIX char *my_strdup(const char * src) {
#if MEM == NVL
	NVL_PREFIX char *dst = nvl_alloc_nv(heap, strlen(src)+1, char);
 	//strcpy(nvl_bare_hack(dst), src);
    char *dst_v = nvl_bare_hack(dst);
    strcpy(dst_v, src);
#if PERSIST
	nvl_persist_hack(dst, strlen(src)+1);
#endif
#elif MEM == VHEAP
	char *dst = nvl_vmalloc(vheap, strlen(src)+1);
    strcpy(dst, src);
#else
	char *dst = (char *)malloc(strlen(src)+1);
    strcpy(dst, src);
#endif
    return dst;	
}

NVL_PREFIX int *my_valdup(int * src, size_t nBytes) {
#if MEM == NVL
	NVL_PREFIX int *dst = nvl_alloc_nv(heap, nBytes/(sizeof(int)), int);
    //memcpy(nvl_bare_hack(dst), src, nBytes);
    int *dst_v = nvl_bare_hack(dst);
    memcpy(dst_v, src, nBytes);
#if PERSIST
	nvl_persist_hack(dst, nBytes);
#endif
#elif MEM == VHEAP
	int *dst = nvl_vmalloc(vheap, nBytes);
    memcpy(dst, src, nBytes);
#else
	int *dst = (int *)malloc(nBytes);
    memcpy(dst, src, nBytes);
#endif
    return dst;	
}


/* Create a new hashtable. */
void ht_create( int size ) {

	int i;

	if( size < 1 ) return;

#if MEM == NVL
    heap = nvl_create(NVLFILE, HEAPSIZE, 0600);
    if(!heap) {
		perror("nvl_create failed");
		exit(1);
    }   
	/* Allocate the table itself. */
#if (MEM == NVL) && !POOR
	if( ( hashtable_nv = nvl_alloc_nv(heap, 1,  hashtable_t ) ) == NULL ) {
        perror("nvl_alloc_nv failed");
		exit(1);
	}
	nvl_set_root(heap, hashtable_nv);

	/* Allocate pointers to the head nodes. */
	if( ( hashtable_nv->table = nvl_alloc_nv(heap, size, nvl entry_t *) ) == NULL ) {
        perror("nvl_alloc_nv failed");
		exit(1);
	}
	hashtable = nvl_bare_hack(hashtable_nv);
#else
	if( ( hashtable = nvl_alloc_nv(heap, 1,  hashtable_t ) ) == NULL ) {
        perror("nvl_alloc_nv failed");
		exit(1);
	}
	nvl_set_root(heap, hashtable);

	/* Allocate pointers to the head nodes. */
	if( ( hashtable->table = nvl_alloc_nv(heap, size, nvl entry_t *) ) == NULL ) {
        perror("nvl_alloc_nv failed");
		exit(1);
	}
#endif
#elif MEM == VHEAP
    vheap = nvl_vcreate(NVLTMP, HEAPSIZE);
    if(!vheap) {
        perror("nvl_vcreate failed");
		exit(1);
    }   
	/* Allocate the table itself. */
	if( ( hashtable = nvl_vmalloc(vheap, sizeof( hashtable_t ) ) ) == NULL ) {
        perror("nvl_vmalloc failed");
		exit(1);
	}

	/* Allocate pointers to the head nodes. */
	if( ( hashtable->table = nvl_vmalloc(vheap, sizeof( entry_t * ) * size ) ) == NULL ) {
        perror("nvl_vmalloc failed");
		exit(1);
	}
	for( i = 0; i < size; i++ ) {
		hashtable->table[i] = NULL;
	}
#else
	/* Allocate the table itself. */
	if( ( hashtable = malloc( sizeof( hashtable_t ) ) ) == NULL ) {
        perror("error in ht_create()");
		exit(1);
	}

	/* Allocate pointers to the head nodes. */
	if( ( hashtable->table = malloc( sizeof( entry_t * ) * size ) ) == NULL ) {
        perror("error in ht_create()");
		exit(1);
	}
	for( i = 0; i < size; i++ ) {
		hashtable->table[i] = NULL;
	}
#endif

	hashtable->size = size;
}

/* Hash a string for a particular hash table. */
int ht_hash( const char *key ) {

	unsigned long int hashval = 0;
	int keyLength = strlen(key);
	int i = 0;

	/* Convert our string to an integer */
#if HASHFUNCTION == 1
	while( hashval < ULONG_MAX && i < keyLength ) {
		hashval = hashval << 8;
		hashval += key[ i ];
		i++;
	}
#else
    for ( i = 0; i < keyLength; ++i ) {
        hashval += key[i], hashval += ( hashval << 10 ), hashval ^= ( hashval >> 6 );
    }
    hashval += ( hashval << 3 ), hashval ^= ( hashval >> 11 ), hashval += ( hashval << 15 );
#endif
	return hashval % hashtable->size;
}

/* Create a key-value pair. */
NVL_PREFIX entry_t *ht_newpair( const char *key, int *value, size_t vSize ) {
	NVL_PREFIX entry_t *newpair = 0;
#if MEM == NVL
	if( ( newpair = nvl_alloc_nv(heap, 1, entry_t) ) == NULL ) {
        perror("error in ht_newpair()");
		exit(1);
	}

	if( ( newpair->key = my_strdup( key ) ) == NULL ) {
        perror("error in ht_newpair()");
		exit(1);
	}

	if( ( newpair->value = my_valdup( value, vSize ) ) == NULL ) {
        perror("error in ht_newpair()");
		exit(1);
	}
	newpair->next = NULL;
#elif MEM == VHEAP
	if( ( newpair = nvl_vmalloc(vheap, sizeof( entry_t ) ) ) == NULL ) {
        perror("error in ht_newpair()");
		exit(1);
	}

	if( ( newpair->key = my_strdup( key ) ) == NULL ) {
        perror("error in ht_newpair()");
		exit(1);
	}

	if( ( newpair->value = my_valdup( value, vSize ) ) == NULL ) {
        perror("error in ht_newpair()");
		exit(1);
	}
	newpair->next = NULL;
#else
	if( ( newpair = malloc( sizeof( entry_t ) ) ) == NULL ) {
        perror("error in ht_newpair()");
		exit(1);
	}

	if( ( newpair->key = my_strdup( key ) ) == NULL ) {
        perror("error in ht_newpair()");
		exit(1);
	}

	if( ( newpair->value = my_valdup( value, vSize ) ) == NULL ) {
        perror("error in ht_newpair()");
		exit(1);
	}
	newpair->next = NULL;
#endif

	return newpair;
}

/* Insert a key-value pair into a hash table. */
void ht_set( const char *key, int *value, size_t vSize ) {
	int bin = 0;
#if MEM == NVL
#if !POOR
	nvl entry_t *newpair_nv = NULL;
	nvl entry_t *next_nv = NULL;
	nvl entry_t *last_nv = NULL;
	entry_t *newpair = NULL;
	entry_t *next = NULL;
	entry_t *last = NULL;
#else
	nvl entry_t *newpair = NULL;
	nvl entry_t *next = NULL;
	nvl entry_t *last = NULL;
#endif
#else
	entry_t *newpair = NULL;
	entry_t *next = NULL;
	entry_t *last = NULL;
#endif
	char *nextkey = NULL;

	bin = ht_hash( key );

#if (MEM == NVL) && !POOR
	next_nv = nvl_nv2nv_to_v2nv_hack(hashtable->table, hashtable_nv)[ bin ];
	next = nvl_bare_hack(next_nv);
#else
	next = hashtable->table[ bin ];
#endif
	if( next != NULL ) {
#if MEM == NVL && !POOR
		nextkey = nvl_bare_hack(nvl_nv2nv_to_v2nv_hack(next->key, next_nv));
#elif MEM == NVL
		nextkey = nvl_bare_hack(next->key);
#else
		nextkey = next->key;
#endif
		nConflicts++;
	}

	while( next != NULL && nextkey != NULL && strcmp( key, nextkey ) > 0 ) {
#if (MEM == NVL) && !POOR
		last_nv = next_nv;
		last = nvl_bare_hack(last_nv);
		next_nv = nvl_nv2nv_to_v2nv_hack(next->next, next_nv);
		next = nvl_bare_hack(next_nv);
#else
		last = next;
		next = next->next;
#endif
		if( next != NULL ) {
#if MEM == NVL && !POOR
			nextkey = nvl_bare_hack(nvl_nv2nv_to_v2nv_hack(next->key, next_nv));
#elif MEM == NVL
			nextkey = nvl_bare_hack(next->key);
#else
			nextkey = next->key;
#endif
		} else {
			nextkey = NULL;
		}
	}

#if TXS
	#pragma nvl atomic heap(heap)
	{
#endif
	/* There's already a pair.  Let's replace that string. */
	if( next != NULL && nextkey != NULL && strcmp( key, nextkey ) == 0 ) {
#if (MEM != NVL) && (MEM != VHEAP)
		free( next->value );
#endif
#if (MEM == NVL) && !POOR
		next_nv->value = my_valdup( value, vSize );
#else
		next->value = my_valdup( value, vSize );
#endif

	/* Nope, could't find it.  Time to grow a pair. */
	} else {
#if (MEM == NVL) && !POOR
		newpair_nv = ht_newpair( key, value, vSize );
		newpair = nvl_bare_hack(newpair_nv);
#else
		newpair = ht_newpair( key, value, vSize );
#endif

#if (MEM == NVL) && !POOR
		/* We're at the start of the linked list in this bin. */
		if( next_nv == nvl_nv2nv_to_v2nv_hack(hashtable->table, hashtable_nv)[ bin ] ) {
			newpair_nv->next = next_nv;
			nvl_nv2nv_to_v2nv_hack(hashtable->table, hashtable_nv)[ bin ] = newpair_nv;
	
		/* We're at the end of the linked list in this bin. */
		} else if ( next == NULL ) {
			last_nv->next = newpair_nv;
	
		/* We're in the middle of the list. */
		} else  {
			newpair_nv->next = next_nv;
			last_nv->next = newpair_nv;
		}
#else
		/* We're at the start of the linked list in this bin. */
		if( next == hashtable->table[ bin ] ) {
			newpair->next = next;
			hashtable->table[ bin ] = newpair;
	
		/* We're at the end of the linked list in this bin. */
		} else if ( next == NULL ) {
			last->next = newpair;
	
		/* We're in the middle of the list. */
		} else  {
			newpair->next = next;
			last->next = newpair;
		}
#endif
	}
#if TXS
	}
#endif
}

/* Retrieve a key-value pair from a hash table. */
NVL_PREFIX int *ht_get( const char *key ) {
	int bin = 0;
#if MEM == NVL
#if !POOR
	NVL_PREFIX entry_t *pair_nv = NULL;
	entry_t *pair = NULL;
#else
	NVL_PREFIX entry_t *pair = NULL;
#endif
#else
	entry_t *pair = NULL;
#endif
	char *pairkey = NULL;

	bin = ht_hash( key );

	/* Step through the bin, looking for our value. */
#if (MEM == NVL) && !POOR
	pair_nv = nvl_nv2nv_to_v2nv_hack(hashtable->table, hashtable_nv)[ bin ];
	pair = nvl_bare_hack(pair_nv);
#else
	pair = hashtable->table[ bin ];
#endif
	if( pair != NULL ) {
#if MEM == NVL && !POOR
		pairkey = nvl_bare_hack(nvl_nv2nv_to_v2nv_hack(pair->key, pair_nv));
#elif MEM == NVL
		pairkey = nvl_bare_hack(pair->key);
#else
		pairkey = pair->key;
#endif
	}
	while( pair != NULL && pairkey != NULL && strcmp( key, pairkey ) > 0 ) {
#if (MEM == NVL) && !POOR
		pair_nv = nvl_nv2nv_to_v2nv_hack(pair->next, pair_nv);
		pair = nvl_bare_hack(pair_nv);
#else
		pair = pair->next;
#endif
		if( pair != NULL ) {
#if MEM == NVL && !POOR
			pairkey = nvl_bare_hack(nvl_nv2nv_to_v2nv_hack(pair->key, pair_nv));
#elif MEM == NVL
			pairkey = nvl_bare_hack(pair->key);
#else
			pairkey = pair->key;
#endif
		}
	}

	/* Did we actually find anything? */
	if( pair == NULL || pairkey == NULL || strcmp( key, pairkey ) != 0 ) {
		return NULL;

	} else {
#if MEM == NVL && !POOR
		return nvl_nv2nv_to_v2nv_hack(pair->value, pair_nv);
#else
		return pair->value;
#endif
	}
	
}

/* Delete a key-value pair from a hash table. */
void ht_del( const char *key ) {
	int bin = 0;
#if MEM == NVL
#if !POOR
	nvl entry_t *newpair_nv = NULL;
	nvl entry_t *next_nv = NULL;
	nvl entry_t *last_nv = NULL;
	entry_t *newpair = NULL;
	entry_t *next = NULL;
	entry_t *last = NULL;
#else
	nvl entry_t *newpair = NULL;
	nvl entry_t *next = NULL;
	nvl entry_t *last = NULL;
#endif
#else
	entry_t *newpair = NULL;
	entry_t *next = NULL;
	entry_t *last = NULL;
#endif
	char *nextkey = NULL;

	bin = ht_hash( key );

#if (MEM == NVL) && !POOR
	next_nv = nvl_nv2nv_to_v2nv_hack(hashtable->table, hashtable_nv)[ bin ];
	next = nvl_bare_hack(next_nv);
#else
	next = hashtable->table[ bin ];
#endif
	if( next != NULL ) {
#if MEM == NVL && !POOR
		nextkey = nvl_bare_hack(nvl_nv2nv_to_v2nv_hack(next->key, next_nv));
#elif MEM == NVL
		nextkey = nvl_bare_hack(next->key);
#else
		nextkey = next->key;
#endif
	}

	while( next != NULL && nextkey != NULL && strcmp( key, nextkey ) > 0 ) {
#if (MEM == NVL) && !POOR
		last_nv = next_nv;
		last = nvl_bare_hack(last_nv);
		next_nv = nvl_nv2nv_to_v2nv_hack(next->next, next_nv);
		//next_nv = next_nv->next;
		next = nvl_bare_hack(next_nv);
#else
		last = next;
		next = next->next;
#endif
		if( next != NULL ) {
#if MEM == NVL && !POOR
			nextkey = nvl_bare_hack(nvl_nv2nv_to_v2nv_hack(next->key, next_nv));
			//nextkey = nvl_bare_hack(next_nv->key);
#elif MEM == NVL
			nextkey = nvl_bare_hack(next->key);
#else
			nextkey = next->key;
#endif
		} else {
			nextkey = NULL;
		}
	}

	/* Found the pair.  Let's delete it. */
#if TXS
	#pragma nvl atomic heap(heap)
#endif
	if( next != NULL && nextkey != NULL && strcmp( key, nextkey ) == 0 ) {
		nDeletes++;
#if (MEM != NVL) && (MEM != VHEAP)
		free( next->value );
#endif

#if (MEM == NVL) && !POOR
		next_nv->key = NULL;
		next_nv->value = NULL;
		nvl entry_t *nextnext = nvl_nv2nv_to_v2nv_hack(next->next, next_nv);
		/* We're at the start of the linked list in this bin. */
		if( next_nv == nvl_nv2nv_to_v2nv_hack(hashtable->table, hashtable_nv)[ bin ] ) {
			if( nextnext != NULL ) {
				nvl_nv2nv_to_v2nv_hack(hashtable->table, hashtable_nv)[ bin ] = nextnext;
			} else {
				nvl_nv2nv_to_v2nv_hack(hashtable->table, hashtable_nv)[ bin ] = NULL;
			}
	
		/* We're at the end of the linked list in this bin. */
		} else if ( nextnext == NULL ) {
			last_nv->next = NULL;
	
		/* We're in the middle of the list. */
		} else  {
			last_nv->next = nextnext;
		}
#else
		next->key = NULL;
		next->value = NULL;
		NVL_PREFIX entry_t *nextnext = next->next;
		/* We're at the start of the linked list in this bin. */
		if( next == hashtable->table[ bin ] ) {
			if( nextnext != NULL ) {
				hashtable->table[ bin ] = nextnext; 
			} else {
				hashtable->table[ bin ] = NULL;
			}
	
		/* We're at the end of the linked list in this bin. */
		} else if ( nextnext == NULL ) {
			last->next = NULL;
	
		/* We're in the middle of the list. */
		} else  {
			last->next = nextnext;
		}
#endif
	}
}

int main( int argc, char **argv ) {
	int i;
	const char *tKey;
	int *tValue;
	int tSum, refSum;
	double time1, time2;
	double time3, time4;

	if( argc == 3 ) { 
		NUMELE = atoi(argv[1]);
		VALSIZE = atoi(argv[2]);
	} else if( argc == 2 ) { 
		NUMELE = atoi(argv[1]);
	}

	printf("NUMELE = %d\n", NUMELE);
	printf("VALSIZE = %d\n", VALSIZE);
#if (MEM == NVL) || (MEM == VHEAP)
	printf("HEAPSIZE = %lu\n", HEAPSIZE);
#endif
#if MEM == NVL
	printf("NVLFILE = %s\n", NVLFILE);
#endif
	ht_create( NUMELE*4 );
	printf("Hashtable is created!\n");
	tValue = (int *)calloc(VALSIZE, sizeof(int));

	printf("Store %d pairs of data with value size of %d!\n", NUMELE, VALSIZE*sizeof(int));
	time1 = my_timer();
	for(i=0; i<NUMELE; i++) {
		tValue[0] = i;
		tKey = my_itoa(i);
		//printf("%d: key = %s, value = %d\n", i, tKey, tValue[0]);
		ht_set( tKey, tValue, VALSIZE*sizeof(int));
	}
	time1 = my_timer() - time1;
	printf("Write time = %lf sec\n", time1);
	printf("# of internal hash conflicts = %d\n", nConflicts);

	printf("Replace %d pairs of data with value size of %d!\n", NUMELE, VALSIZE*sizeof(int));
	time2 = my_timer();
	for(i=0; i<NUMELE; i++) {
		tValue[0] = NUMELE-i-1;
		tKey = my_itoa(i);
		//printf("%d: key = %s, value = %d\n", i, tKey, tValue[0]);
		ht_set( tKey, tValue, VALSIZE*sizeof(int));
	}
	time2 = my_timer() - time2;
	printf("Replace time = %lf sec\n", time2);

	printf("Retrieve %d pairs of data with value size of %d!\n", NUMELE, VALSIZE*sizeof(int));
	time3 = my_timer();
	tSum = 0;
	refSum = 0;
	for(i=0; i<NUMELE; i++) {
		tKey = my_itoa(i);
		refSum += i;
		//printf("key = %s, value = %d\n", tKey, *(ht_get(tKey)));
		tSum += *(ht_get(tKey));
	}
	time3 = my_timer() - time3;
	printf("Found %d records\n", i);
	printf("Read time = %lf sec\n", time3);

	printf("Delete %d pairs of data!\n", NUMELE);
	time4 = my_timer();
	for(i=0; i<NUMELE; i++) {
		tKey = my_itoa(i);
		//printf("%d: key = %s, value = %d\n", i, tKey, tValue[0]);
		ht_del(tKey);
	}
	time4 = my_timer() - time4;
	printf("Delete time = %lf sec\n", time4);
	printf("# of deletes= %d\n", nDeletes);

	printf("Write+Replace+Read+Delete time = %lf sec\n", time1 + time2 + time3 + time4);

	if( refSum == tSum ) {
		printf("Verification passed!\n");
	} else {
		printf("Verification failed!\n");
	}

	return 0;
}

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0);

    return time.tv_sec + time.tv_usec / 1000000.0;
}

const char *my_itoa_buf(char *buf, size_t len, int num)
{
  static char loc_buf[sizeof(int) * CHAR_BITS]; /* not thread safe */

  if (!buf)
  {
    buf = loc_buf;
    len = sizeof(loc_buf);
  }

  if (snprintf(buf, len, "%d", num) == -1)
    return ""; /* or whatever */

  return buf;
}

const char *my_itoa(int num)
{ return my_itoa_buf(NULL, 0, num); }

