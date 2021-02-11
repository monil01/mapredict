AC_DEFUN([ACX_BLAS], [
	AC_CHECK_LIB([gfortran], [_gfortran_st_write_done],
		[
			PLASMA_comLIBS="-lgfortran ${PLASMA_comLIBS}"
			LIBS="$LIBS -lgfortran"
		],
		[
			AC_MSG_WARN([couldn't find _gfortran_st_write_done in -lgfortran. Deactivating compilation of the PLASMA benchmarks.])
dnl '
			KASTORS_COMPILE_PLASMA=no
			KASTORS_MISSING_DEPS="$KASTORS_MISSING_DEPS -lgfortran"
		])

	AC_CHECK_LIB([blas], [dgemm_],
		[
			PLASMA_comLIBS="-lblas ${PLASMA_comLIBS}"
			LIBS="$LIBS -lblas"
		],
		[
			AC_MSG_WARN([couldn't find dgemm_ in -lblas. Deactivating compilation of the PLASMA benchmarks.])
dnl '
			KASTORS_COMPILE_PLASMA=no
			KASTORS_MISSING_DEPS="$KASTORS_MISSING_DEPS -lblas"
		])

	AC_CHECK_LIB([lapack], [dlacpy_],
		[
			PLASMA_comLIBS="-llapack ${PLASMA_comLIBS}"
			LIBS="$LIBS -llapack"
		],
		[
			AC_MSG_WARN([couldn't find dlacpy_ in -llapack. Deactivating compilation of the PLASMA benchmarks.])
dnl '
			KASTORS_COMPILE_PLASMA=no
			KASTORS_MISSING_DEPS="$KASTORS_MISSING_DEPS -llapack"
		])

	ACX_LAPACKE([$1])
])
