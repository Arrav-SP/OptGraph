#include <stdio.h>
#define N 96
/* Histogram + prefix sums + branchy post-processing. */
int main(void) {
    unsigned data[N], hist[8] = {0u}, prefix[N];
    unsigned seed = 12345u;
    for (int i = 0; i < N; i++) {
        seed = seed * 1103515245u + 12345u;
        data[i] = (seed >> 16) & 0xffu;
    }
    for (int i = 0; i < N; i++) hist[data[i] >> 5]++;
    prefix[0] = data[0];
    for (int i = 1; i < N; i++) prefix[i] = prefix[i - 1] + data[i];
    unsigned best = 0u, odd = 0u;
    for (int i = 0; i < 8; i++) if (hist[i] > hist[best]) best = (unsigned)i;
    for (int i = 0; i < N; i++) if (prefix[i] & 1u) odd++; else odd += 0u;
    printf("%u %u %u\n", best, odd, prefix[N - 1]);
    return 0;
}