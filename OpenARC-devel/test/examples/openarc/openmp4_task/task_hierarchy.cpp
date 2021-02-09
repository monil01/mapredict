#include <stdio.h>
#include "omp_helper.h"
#include "omp_helper_ext.h"

#define SIZE 1024

int main(int argc, char** argv) {
    int A[SIZE];
    int B[SIZE];
    int C[SIZE];
    int D[SIZE];

    omp_helper_set_queue_max(4);
    omp_helper_set_queue_off(0);

#pragma omp parallel num_threads(4)
    {
#pragma omp single nowait
    {
    {
        int arg0[3] = { oh_out, oh_out, oh_out };
        void* arg1[6] = { &A[0], &A[SIZE - 1], &B[0], &B[SIZE - 1], &C[0], &C[SIZE - 1] };
        omp_helper_task_enter(3, arg0, arg1);
    }
#pragma omp task depend(out:A,B,C)
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
    }
    {
        omp_helper_task_exit();
    }

    {
        int arg0[4] = { oh_in, oh_in, oh_inout, oh_out };
        void* arg1[8] = { &A[0], &A[SIZE - 1], &B[0], &B[SIZE - 1], &C[0], &C[SIZE - 1], &D[0], &D[SIZE - 1] };
        omp_helper_task_enter(4, arg0, arg1);
    }
#pragma omp task depend(in:A,B) depend(inout:C) depend(out:D)
    {

    {
        int arg0[1] = { oh_out };
        void* arg1[2] = { &C[0], &C[SIZE - 1] };
        int async;
        int waits[4];
        omp_helper_task_exec(1, arg0, arg1, &async, waits);
        printf("[%s:%d] async[%d] waits[%d,%d,%d,%d]\n", __FILE__, __LINE__, async, waits[0], waits[1], waits[2], waits[3]);
    }
#pragma omp task depend(inout:C)
    for (int i = 0; i < SIZE; i++) {
        C[i] += 4 * (A[i] + B[i]);
    }

    {
        int arg0[1] = { oh_out };
        void* arg1[2] = { &D[0], &D[SIZE - 1] };
        int async;
        int waits[4];
        omp_helper_task_exec(1, arg0, arg1, &async, waits);
        printf("[%s:%d] async[%d] waits[%d,%d,%d,%d]\n", __FILE__, __LINE__, async, waits[0], waits[1], waits[2], waits[3]);
    }
#pragma omp task depend(out:D)
    for (int i = 0; i < SIZE; i++) {
        D[i] = i * 10;
    }
    }
    {
        omp_helper_task_exit();
    }

    {
        int arg0[2] = { oh_in, oh_out };
        void* arg1[4] = { &A[0], &A[SIZE - 1], &D[0], &D[SIZE - 1] };
        omp_helper_task_enter(2, arg0, arg1);
    }
#pragma omp task depend(in:A) depend(out:D)
    {

    {
        int arg0[2] = { oh_in, oh_out };
        void* arg1[4] = { &A[0], &A[SIZE - 1], &D[0], &D[SIZE - 1] };
        int async;
        int waits[4];
        omp_helper_task_exec(2, arg0, arg1, &async, waits);
        printf("[%s:%d] async[%d] waits[%d,%d,%d,%d]\n", __FILE__, __LINE__, async, waits[0], waits[1], waits[2], waits[3]);
    }
#pragma omp task depend(in:A) depend(out:D)
    for (int i = 0; i < SIZE; i++) {
        D[i] = A[i] * 5;
    }

    {
        int arg0[2] = { oh_in, oh_in };
        void* arg1[4] = { &A[0], &A[SIZE - 1], &D[0], &D[SIZE - 1] };
        int async;
        int waits[4];
        omp_helper_task_exec(2, arg0, arg1, &async, waits);
        printf("[%s:%d] async[%d] waits[%d,%d,%d,%d]\n", __FILE__, __LINE__, async, waits[0], waits[1], waits[2], waits[3]);
    }
#pragma omp task depend(in:A,D)
    for (int i = 0; i < SIZE; i++) {
        if (D[i] != 5 * A[i])
            printf("X [%4d] %4d = 5 * %4d\n", i, D[i], A[i]);
    }
    }
    {
        omp_helper_task_exit();
    }
    }
    }

    return 0;
}

