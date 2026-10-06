#include <stdio.h>
#define N 200
/* Element-wise copy, reverse copy, then compare. */
int main(void) {
    int src[N], dst[N], rev[N];
    for (int i = 0; i < N; i++) src[i] = i * 3 - 50;
    for (int i = 0; i < N; i++) dst[i] = src[i];
    for (int i = 0; i < N; i++) rev[i] = src[N - 1 - i];
    int same = 0, cross = 0;
    for (int i = 0; i < N; i++) {
        if (dst[i] == src[i]) same++;
        if (rev[i] == dst[i]) cross++;
    }
    printf("%d %d\n", same, cross);
    return 0;
}