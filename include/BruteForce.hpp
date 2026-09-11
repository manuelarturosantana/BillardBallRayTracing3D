#include <optional>
#include <vector>
#include "ScatObject.hpp"

// No tree, no pruning — just test every object. This is the ground
// truth the BVH's answers get checked against.
inline std::optional<Hit> brute_force_intersect(const Ray& ray,
                                                  const std::vector<const ScatObject*>& objects) {
    std::optional<Hit> best;
    for (auto* obj : objects) {
        auto hit = obj->intersect(ray);
        if (hit && (!best || hit->t < best->t))
            best = hit;
    }
    return best;
}