#include <stdio.h>
#define N 12
/* Dense integer matrix multiplication and trace. */
int main(void) {
    unsigned a[N][N], b[N][N], c[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            a[i][j] = (unsigned)(i + j);
            b[i][j] = (unsigned)(i * j + 1);
        }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            c[i][j] = 0u;
            for (int k = 0; k < N; k++) c[i][j] += a[i][k] * b[k][j];
        }
    unsigned tr = 0u;
    for (int i = 0; i < N; i++) tr += c[i][i];
    printf("%u\n", tr);
    return 0;
}