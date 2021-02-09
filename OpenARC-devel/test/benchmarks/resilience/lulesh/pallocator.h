/* Register a block as persistent memory */
int perm(void *ptr, size_t size);

/* Open and map file into core memory */
int mopen(const char *fname, const char *mode, size_t size);

/* Close memory-mapped file */
int mclose(void);

/* Flushes in-core data to memory-mapped file */
int mflush(void);

/* Open backup file */
int bopen(const char *fname, const char *mode);

/* Close backup file */
int bclose(void);

/* Backup globals and heap to a separate file */
int backup(void);

/* Restore globals and heap from a separate file */
int restore(void);
