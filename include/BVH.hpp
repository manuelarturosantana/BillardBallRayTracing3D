#pragma once

#include <vector>
#include <algorithm>
#include <optional>
#include <limits>
#include <cstdint>
#include "ScatObject.hpp"   // Vec3, Ray, AABB, Hit, ScatObject

// --- Ray/box test -----------------------------------------------------
// Slab method, taking precomputed inverse direction so callers doing
// many box tests per ray (i.e. every traversal) don't repeat the
// division. Returns false on a miss.
inline bool ray_box(const Vec3& origin, const double inv_dir[3],
                     const AABB& box, double t_hit_so_far,
                     double& tmin_out) {
    double tmin = -std::numeric_limits<double>::infinity();
    double tmax =  std::numeric_limits<double>::infinity();

    const double o[3]    = { origin.x, origin.y, origin.z };
    const double bmin[3] = { box.min.x, box.min.y, box.min.z };
    const double bmax[3] = { box.max.x, box.max.y, box.max.z };

    for (int axis = 0; axis < 3; ++axis) {
        double t1 = (bmin[axis] - o[axis]) * inv_dir[axis];
        double t2 = (bmax[axis] - o[axis]) * inv_dir[axis];
        if (inv_dir[axis] < 0.0) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        if (tmax < tmin) return false;
    }

    if (tmin > t_hit_so_far) return false;   // prune: box is entirely
                                              // farther than best hit found
    tmin_out = tmin;
    return tmax >= std::max(tmin, 0.0);
}

inline AABB union_box(const AABB& a, const AABB& b) {
    AABB result;
    result.min = { std::min(a.min.x, b.min.x),
                    std::min(a.min.y, b.min.y),
                    std::min(a.min.z, b.min.z) };
    result.max = { std::max(a.max.x, b.max.x),
                    std::max(a.max.y, b.max.y),
                    std::max(a.max.z, b.max.z) };
    return result;
}

// --- Flattened tree node --------------------------------------------------
// Depth-first array layout. Internal node: left child is implicitly
// this_index + 1 (laid out immediately after its parent); right_child
// is stored explicitly since it isn't adjacent. Leaf: obj_start/obj_count
// index into a separate flat object-index array (avoids a
// vector-per-leaf allocation).
struct alignas(32) BVHNode {
    AABB    box;
    int32_t right_child;   // internal only; -1 marks a leaf
    int32_t obj_start;     // leaf only: offset into BVH::leaf_objects_
    int32_t obj_count;     // leaf only: 0 for internal nodes
    uint8_t split_axis;    // internal only: 0/1/2 for x/y/z, used for near/far order
};

// --- BVH ------------------------------------------------------------------
class BVH {
public:
    explicit BVH(std::vector<const ScatObject*> objects, int leaf_threshold = 4)
        : leaf_threshold_(leaf_threshold)
    {
        nodes_.reserve(2 * objects.size());   // upper bound for a binary tree
        leaf_objects_.reserve(objects.size());
        build(std::move(objects));
    }

    std::optional<Hit> intersect(const Ray& ray) const {
        // Precompute once per ray, not once per node visited.
        double inv_dir[3] = {
            1.0 / ray.dir.x, 1.0 / ray.dir.y, 1.0 / ray.dir.z
        };
        bool dir_neg[3] = { inv_dir[0] < 0.0, inv_dir[1] < 0.0, inv_dir[2] < 0.0 };

        std::optional<Hit> best;
        double best_t = std::numeric_limits<double>::infinity();

        // Explicit stack, sized generously for tree depth (O(log N) in
        // practice; 64 covers any realistic patch count many times over).
        int32_t stack[64];
        int sp = 0;
        stack[sp++] = 0;   // root is always node 0 in depth-first layout

        while (sp > 0) {
            int32_t idx = stack[--sp];
            const BVHNode& node = nodes_[idx];

            double tmin;
            if (!ray_box(ray.origin, inv_dir, node.box, best_t, tmin))
                continue;

            if (node.right_child < 0) {
                // Leaf: test every object directly.
                for (int32_t i = 0; i < node.obj_count; ++i) {
                    const ScatObject* obj = leaf_objects_[node.obj_start + i];
                    auto hit = obj->intersect(ray);
                    if (hit && hit->t < best_t) {
                        best_t = hit->t;
                        hit->object = obj;   // tag with the object that produced it
                        best = hit;
                    }
                }
                continue;
            }

            // Internal: push far child first, near child second, so the
            // near child is popped and visited first. "Near" is decided
            // by the sign of the ray direction along this node's split
            // axis, not by an extra box test.
            int32_t left = idx + 1;                 // implicit left child
            int32_t right = node.right_child;
            if (dir_neg[node.split_axis]) {
                stack[sp++] = left;
                stack[sp++] = right;
            } else {
                stack[sp++] = right;
                stack[sp++] = left;
            }
        }
        return best;
    }

private:
    std::vector<BVHNode> nodes_;
    std::vector<const ScatObject*> leaf_objects_;
    int leaf_threshold_;

    // Returns the index of the node just built.
    int32_t build(std::vector<const ScatObject*> objects) {
        int32_t this_idx = static_cast<int32_t>(nodes_.size());
        nodes_.emplace_back();   // reserve the slot now so children of
                                  // this node land after it (needed for
                                  // the implicit-left-child layout)

        AABB box = objects.front()->bounding_box();
        for (size_t i = 1; i < objects.size(); ++i)
            box = union_box(box, objects[i]->bounding_box());

        if (static_cast<int>(objects.size()) <= leaf_threshold_) {
            nodes_[this_idx].box = box;
            nodes_[this_idx].right_child = -1;
            nodes_[this_idx].obj_start = static_cast<int32_t>(leaf_objects_.size());
            nodes_[this_idx].obj_count = static_cast<int32_t>(objects.size());
            for (auto* obj : objects) leaf_objects_.push_back(obj);
            return this_idx;
        }

        Vec3 c_min = objects.front()->centroid();
        Vec3 c_max = c_min;
        for (auto* obj : objects) {
            const Vec3& c = obj->centroid();
            c_min.x = std::min(c_min.x, c.x); c_max.x = std::max(c_max.x, c.x);
            c_min.y = std::min(c_min.y, c.y); c_max.y = std::max(c_max.y, c.y);
            c_min.z = std::min(c_min.z, c.z); c_max.z = std::max(c_max.z, c.z);
        }
        double spread[3] = { c_max.x - c_min.x, c_max.y - c_min.y, c_max.z - c_min.z };
        uint8_t axis = static_cast<uint8_t>(
            std::distance(spread, std::max_element(spread, spread + 3)));

        auto mid = objects.begin() + objects.size() / 2;
        std::nth_element(objects.begin(), mid, objects.end(),
            [axis](const ScatObject* a, const ScatObject* b) {
                const Vec3& ca = a->centroid();
                const Vec3& cb = b->centroid();
                double va = (axis == 0) ? ca.x : (axis == 1) ? ca.y : ca.z;
                double vb = (axis == 0) ? cb.x : (axis == 1) ? cb.y : cb.z;
                return va < vb;
            });

        std::vector<const ScatObject*> left_objs(objects.begin(), mid);
        std::vector<const ScatObject*> right_objs(mid, objects.end());
        objects.clear(); objects.shrink_to_fit();   // release before recursing

        build(std::move(left_objs));                 // becomes this_idx + 1
        int32_t right_idx = build(std::move(right_objs));

        nodes_[this_idx].box = box;
        nodes_[this_idx].right_child = right_idx;
        nodes_[this_idx].obj_count = 0;
        nodes_[this_idx].split_axis = axis;
        return this_idx;
    }
};