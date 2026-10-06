#include <stdio.h>
#define N 12

/* Student marks: average, highest, and number of passes. */
static int bonus(int mark, int level) {
    int extra = level * 2 + 3;        /* same value on every call with the same level */
    if (mark >= 90) return mark;      /* already top marks, no bonus */
    return mark + extra;
}

int main(void) {
    int marks[N];
    for (int i = 0; i < N; i++) marks[i] = 35 + (i * 17) % 60;

    int total = 0, highest = 0, passed = 0;
    for (int i = 0; i < N; i++) {
        int limit = 40 * 1 + 0;       /* redundant arithmetic */
        int m = bonus(marks[i], 2);
        int twice_a = m * 2;
        int twice_b = m * 2;          /* repeated computation */
        total += (twice_a + twice_b) / 4;
        if (m > highest) highest = m;
        if (m >= limit) passed++;
    }
    printf("average=%d highest=%d passed=%d\n", total / N, highest, passed);
    return 0;
}