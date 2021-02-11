/* $Id: config.h,v 1.5 2004/04/20 14:42:56 pohlt Exp $ */

/*############################################################################*/

#ifndef _CONFIG_H_
#define _CONFIG_H_

/*############################################################################*/

#define SIZE   (100)
#define SIZE_X (1*SIZE)
#define SIZE_Y (1*SIZE)
#define SIZE_Z (130)

#ifdef _OPENARC_
#pragma openarc #define SIZE   (100)
#pragma openarc #define SIZE_X (1*SIZE)
#pragma openarc #define SIZE_Y (1*SIZE)
#pragma openarc #define SIZE_Z (130)
#endif

#define OMEGA (1.95)

#define OUTPUT_PRECISION float

#define BOOL int
#define TRUE (-1)
#define FALSE (0)

/*############################################################################*/

#endif /* _CONFIG_H_ */
