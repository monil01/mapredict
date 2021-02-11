AC_DEFUN([ACX_LAPACKE], [
	AC_CHECK_HEADER([lapacke.h], [],
		[
			AC_MSG_WARN([couldn't find lapacke.h header. Deactivating compilation of the PLASMA benchmarks.])
dnl '
			KASTORS_COMPILE_PLASMA=no
			KASTORS_MISSING_DEPS="$KASTORS_MISSING_DEPS lapacke.h"
		])

	AS_IF( [ test "x$1" == "x" ], [LAPACKE_LIB="lapacke"], [LAPACKE_LIB="$1"])

	AC_CHECK_LIB([${LAPACKE_LIB}], [LAPACKE_dlacpy_work],
		[
			AS_IF([ test "x$1" == "x" ], [], [
				PLASMA_comLIBS="${PLASMA_comLIBS}"
				LIBS="$LIBS"
			])
		], [
			AC_MSG_WARN([couldn't find LAPACKE_dlacpy_work in -l${LAPACKE_LIB}. Deactivating compilation of the PLASMA benchmarks.])
dnl '
			KASTORS_COMPILE_PLASMA=no
			KASTORS_MISSING_DEPS="$KASTORS_MISSING_DEPS ${LAPACKE_LIB}"
		])
])
