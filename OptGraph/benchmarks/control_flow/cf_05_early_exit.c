#include <stdio.h>
#define N 100
/* Searches with break/continue and multiple return points. */
static int find_first(const int *a, int n, int key) {
    for (int i = 0; i < n; i++) {
        if (a[i] < 0) continue;
        if (a[i] == key) return i;
    }
    return -1;
}
int main(void) {
    int a[N];
    for (int i = 0; i < N; i++) a[i] = (i % 9 == 4) ? -1 : (i * 13) % 71;
    int hits = 0, pos = 0;
    for (int k = 0; k < 80; k++) {
        int idx = find_first(a, N, k);
        if (idx < 0) continue;
        hits++;
        pos += idx;
        if (pos > 2000) break;
    }
    printf("%d %d\n", hits, pos);
    return 0;
}