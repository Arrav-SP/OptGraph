#include <stdio.h>
#define N 256
/* Fill an array, then sum it. */
int main(void) {
    unsigned a[N];
    for (int i = 0; i < N; i++) a[i] = (unsigned)(i * 7 + 3);
    unsigned s = 0u;
    for (int i = 0; i < N; i++) s += a[i];
    printf("%u\n", s);
    return 0;
}