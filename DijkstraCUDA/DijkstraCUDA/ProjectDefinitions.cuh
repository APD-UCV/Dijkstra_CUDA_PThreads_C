
#ifndef PROJECTDEFINITIONS_H
#define PROJECTDEFINITIONS_H

/* Standard library imports */
#include <stdio.h>

/*Project core specific macros*/
#define CONST const
#define CONSTVAR(dataType, memClass) CONST dataType
#define VAR(dataType, memClass) dataType

#define EXTERN extern
#define STATIC static

#define HOST 
#define DEVICE 
#define GLOBAL __global__

#define P2CONST(ptrType, memClass) CONST ptrType* __restrict__
#define CONSTP2VAR(ptrType, memClass) ptrType* CONST __restrict__
#define P2VAR(ptrType, memClass) ptrType* __restrict__

#define FUNC(dataType, memClass) memClass dataType
#define P2FUNC(retType, memClass, funcName) memClass retType (*funcName)

#define STD_RETURN_TYPE int 

#define STD_OK 1
#define STD_NOT_OK 0

#define MARKED 1
#define NOT_MARKED 0

#define CUDA_SAFE_CALL(METHOD) { gpuErrchk((METHOD), __FILE__, __LINE__); }
inline void gpuErrchk(cudaError_t code, P2CONST(char, HOST) file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#endif // !PROJECTDEFINITIONS_H
