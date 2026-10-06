#include <stdio.h>
#define R 16
#define C 24
/* Initialise 1-D and 2-D arrays with different patterns, with redundant stores. */
int main(void) {
    unsigned grid[R][C];
    unsigned row[C];
    for (int j = 0; j < C; j++) { row[j] = 0u; row[j] = (unsigned)(j * j); }
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            grid[i][j] = 0u;
            grid[i][j] = row[j] + (unsigned)i;
        }
    unsigned s = 0u;
    for (int i = 0; i < R; i++) s += grid[i][(i * 5) % C];
    printf("%u\n", s);
    return 0;
}