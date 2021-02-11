#define AOCL_ALIGNMENT 64

// Directive versions
#define DIRECTIVE         0
#define DIRECTIVE_MULTI   1

// Manual versions
#define ND_RANGE          0
#define SINGLE_WORK_ITEM  0
#define SW_BASE           0
#define SW_MULTI          0

// Transplant versions
#define ALTERA            0

#define ROWS 1080
#ifdef _OPENARC_
#pragma openarc #define ROWS 1080
#endif

#define COLS 1920
#ifdef _OPENARC_
#pragma openarc #define COLS 1920 
#endif

#ifndef MAX 
#ifndef __cplusplus
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif
#endif

#ifndef MIN 
#ifndef __cplusplus
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif
#endif

#ifndef  __cplusplus
#define abs(x) ((x)<0 ? -(x) : (x))
#endif
