#include <stdio.h>
/* Small aggregates passed and returned by value: scalar replacement targets. */
struct vec3 { int x, y, z; };
struct box { struct vec3 lo, hi; };
static struct vec3 add(struct vec3 a, struct vec3 b) {
    struct vec3 r;
    r.x = a.x + b.x; r.y = a.y + b.y; r.z = a.z + b.z;
    return r;
}
static int volume(struct box b) {
    struct vec3 d;
    d.x = b.hi.x - b.lo.x; d.y = b.hi.y - b.lo.y; d.z = b.hi.z - b.lo.z;
    return d.x * d.y * d.z;
}
int main(void) {
    struct box b;
    b.lo.x = 0; b.lo.y = 1; b.lo.z = 2;
    b.hi.x = 4; b.hi.y = 6; b.hi.z = 8;
    struct vec3 step = {1, 1, 2};
    int total = 0;
    for (int i = 0; i < 10; i++) {
        total += volume(b);
        b.hi = add(b.hi, step);
    }
    printf("%d\n", total);
    return 0;
}