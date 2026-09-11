#pragma once

#include "../ScatObject.hpp"
#include "../Utils/MollerTrumbore.hpp"
#include <algorithm>

// This triangle class is for testing the methods.

class Triangle : public ScatObject {
public:
    Triangle(Vec3 v0, Vec3 v1, Vec3 v2) : v0_(v0), v1_(v1), v2_(v2) {
        // Bounding box: exact for a flat triangle, no inflation needed
        // (unlike the curved-patch case) — three points, min/max, done.
        box_.min = { std::min({v0.x, v1.x, v2.x}),
                     std::min({v0.y, v1.y, v2.y}),
                     std::min({v0.z, v1.z, v2.z}) };
        box_.max = { std::max({v0.x, v1.x, v2.x}),
                     std::max({v0.y, v1.y, v2.y}),
                     std::max({v0.z, v1.z, v2.z}) };

        centroid_ = { (v0.x + v1.x + v2.x) / 3.0,
                      (v0.y + v1.y + v2.y) / 3.0,
                      (v0.z + v1.z + v2.z) / 3.0 };
    }

    std::optional<Hit> intersect(const Ray& ray) const override {
        return moller_trumbore(ray, v0_, v1_, v2_);
    }

private:
    Vec3 v0_, v1_, v2_;
};
