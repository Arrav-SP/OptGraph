#include <stdio.h>
#define N 500
/* Sieve of Eratosthenes followed by a twin-prime count. */
int main(void) {
    unsigned char composite[N + 1];
    for (int i = 0; i <= N; i++) composite[i] = 0;
    composite[0] = composite[1] = 1;
    for (int p = 2; p * p <= N; p++) {
        if (composite[p]) continue;
        for (int m = p * p; m <= N; m += p) composite[m] = 1;
    }
    int count = 0, twins = 0, last = -10;
    for (int i = 2; i <= N; i++) {
        if (composite[i]) continue;
        count++;
        if (i - last == 2) twins++;
        last = i;
    }
    printf("%d %d\n", count, twins);
    return 0;
}