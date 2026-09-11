#include <iostream>
#include <cassert>
#include "../include/ScatObjects/Triangle.hpp"
#include "../include/BVH.hpp"
#include "../include/BruteForce.hpp"

bool nearly_equal(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

int main() {
    // A handful of triangles spread out in space, so the BVH actually
    // has to split them rather than putting everything in one leaf.
    std::vector<Triangle> tris;
    tris.emplace_back(Vec3{0,0,0},  Vec3{1,0,0},  Vec3{0,1,0});   // near origin, z=0 plane
    tris.emplace_back(Vec3{10,0,0}, Vec3{11,0,0}, Vec3{10,1,0});  // far away along x
    tris.emplace_back(Vec3{0,0,5},  Vec3{1,0,5},  Vec3{0,1,5});   // shifted in z
    tris.emplace_back(Vec3{-5,0,0}, Vec3{-4,0,0}, Vec3{-5,1,0});  // far along -x

    std::vector<const ScatObject*> objects;
    for (auto& t : tris) objects.push_back(&t);

    BVH bvh(objects, /*leaf_threshold=*/1);   // small threshold to force real splitting

    struct TestRay { Ray ray; const char* label; };
    std::vector<TestRay> test_rays = {
        { { {0.2, 0.2, -1}, {0, 0, 1} },  "should hit first triangle at t=1" },
        { { {10.2, 0.2, -1}, {0, 0, 1} }, "should hit far triangle at t=1" },
        { { {100, 100, -1}, {0, 0, 1} },  "should miss everything" },
        { { {0.2, 0.2, 10}, {0, 0, -1} }, "should hit triangle at z=5 first (t=5), not z=0" },
    };

    int failures = 0;
    for (auto& tr : test_rays) {
        auto bvh_hit    = bvh.intersect(tr.ray);
        auto brute_hit  = brute_force_intersect(tr.ray, objects);

        bool agree = (bvh_hit.has_value() == brute_hit.has_value()) &&
                     (!bvh_hit || nearly_equal(bvh_hit->t, brute_hit->t));

        std::cout << tr.label << ": "
                  << (agree ? "PASS" : "FAIL")
                  << " (bvh t=" << (bvh_hit ? bvh_hit->t : -1)
                  << ", brute t=" << (brute_hit ? brute_hit->t : -1) << ")\n";

        if (!agree) failures++;
    }

    std::cout << (failures == 0 ? "\nAll tests passed.\n" : "\nSome tests FAILED.\n");
    return failures;
}