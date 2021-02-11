#include <stdio.h>
#include "omp_helper.h"

#define SIZE 1024

int main(int argc, char** argv) {
    int A[SIZE];
    int B[SIZE];
    int C[SIZE];

    omp_helper_set_queue_max(4);
    omp_helper_set_queue_off(0);

#pragma omp parallel num_threads(4)
    {
#pragma omp single nowait
    {

    {
        int arg0[1] = { oh_out };
        void* arg1[2] = { &A[0], &A[SIZE - 1] };
        int async;
        int waits[4];
        omp_helper_task_exec(1, arg0, arg1, &async, waits);
        printf("[%s:%d] async[%d] waits[%d,%d,%d,%d]\n", __FILE__, __LINE__, async, waits[0], waits[1], waits[2], waits[3]);
    }
#pragma omp task depend(out:A)
    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
    }

    {
        int arg0[1] = { oh_out };
        void* arg1[2] = { &B[0], &B[SIZE - 1] };
        int async;
        int waits[4];
        omp_helper_task_exec(1, arg0, arg1, &async, waits);
        printf("[%s:%d] async[%d] waits[%d,%d,%d,%d]\n", __FILE__, __LINE__, async, waits[0], waits[1], waits[2], waits[3]);
    }
#pragma omp task depend(out:B)
    for (int i = 0; i < SIZE; i++) {
        B[i] = i * 10;
    }

    {
        int arg0[3] = { oh_in, oh_in, oh_out };
        void* arg1[6] = { &A[0], &A[SIZE - 1], &B[0], &B[SIZE - 1], &C[0], &C[SIZE - 1] };
        int async;
        int waits[4];
        omp_helper_task_exec(3, arg0, arg1, &async, waits);
        printf("[%s:%d] async[%d] waits[%d,%d,%d,%d]\n", __FILE__, __LINE__, async, waits[0], waits[1], waits[2], waits[3]);
    }
#pragma omp task depend(in:A,B) depend(out:C)
    for (int i = 0; i < SIZE; i++) {
        C[i] = A[i] + B[i];
    }

    {
        int arg0[1] = { oh_inout };
        void* arg1[2] = { &C[0], &C[SIZE - 1] };
        int async;
        int waits[4];
        omp_helper_task_exec(1, arg0, arg1, &async, waits);
        printf("[%s:%d] async[%d] waits[%d,%d,%d,%d]\n", __FILE__, __LINE__, async, waits[0], waits[1], waits[2], waits[3]);
    }
#pragma omp task depend(inout:C)
    for (int i = 0; i < SIZE; i++) {
        C[i] += A[i] + B[i];
    }

    {
        int arg0[1] = { oh_in };
        void* arg1[2] = { &C[0], &C[SIZE - 1] };
        int async;
        int waits[4];
        omp_helper_task_exec(1, arg0, arg1, &async, waits);
        printf("[%s:%d] async[%d] waits[%d,%d,%d,%d]\n", __FILE__, __LINE__, async, waits[0], waits[1], waits[2], waits[3]);
    }
#pragma omp task depend(in:C)
    for (int i = 0; i < SIZE; i++) {
        if (C[i] != 2 * (A[i] + B[i]))
            printf("X [%4d] %4d = %4d + %4d\n", i, C[i], A[i], B[i]);
    }

    }
    }

    return 0;
}

