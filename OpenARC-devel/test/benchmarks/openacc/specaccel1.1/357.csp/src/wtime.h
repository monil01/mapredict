/* C/Fortran interface is different on different machines. 
 * You may need to tweak this.
 */

/* C/Fortran interface is not needed for 352.ep */
#ifndef SPEC
#if defined(IBM)
#define wtime wtime
#elif defined(CRAY)
#define wtime WTIME
#else
#define wtime wtime_
#endif
#endif
