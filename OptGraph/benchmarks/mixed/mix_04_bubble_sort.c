#include <stdio.h>
#define N 48
/* Bubble sort with an early-exit flag, then a sortedness check. */
static void swap(int *x, int *y) { int t = *x; *x = *y; *y = t; }
int main(void) {
    int a[N];
    for (int i = 0; i < N; i++) a[i] = (i * 29 + 11) % 53;
    for (int pass = 0; pass < N - 1; pass++) {
        int swapped = 0;
        for (int j = 0; j < N - 1 - pass; j++)
            if (a[j] > a[j + 1]) { swap(&a[j], &a[j + 1]); swapped = 1; }
        if (!swapped) break;
    }
    int ok = 1, chk = 0;
    for (int i = 0; i < N; i++) {
        if (i > 0 && a[i - 1] > a[i]) ok = 0;
        chk += a[i] * (i + 1);
    }
    printf("%d %d\n", ok, chk);
    return 0;
}