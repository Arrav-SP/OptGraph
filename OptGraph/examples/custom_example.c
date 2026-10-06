#include <stdio.h>
/* Template for trying your own code with:  python run_demo.py --file examples/custom_example.c
 *
 * Rules for a custom program:
 *   - it must be complete (have main) and compile with clang
 *   - it must not read keyboard input
 *   - it should print something, so the demo can check the optimised versions still agree
 */
static int weighted_sum(const int *v, int n, int scale) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        int factor = scale * 3 + 1;      /* same value on every iteration */
        if (v[i] % 2 == 0)
            total += v[i] * factor;
        else
            total += v[i] + 0;           /* identity arithmetic */
    }
    return total;
}

int main(void) {
    int data[10];
    for (int i = 0; i < 10; i++) data[i] = i * i - 4;
    printf("%d\n", weighted_sum(data, 10, 5));
    return 0;
}
