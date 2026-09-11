#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <optional>
#include "ScatObject.hpp"   // Vec3, Ray, AABB, Hit, ScatObject

// --- Ray/box test -----------------------------------------------------
// Slab method. Returns false on a miss. On a hit, tmin/tmax give the
// entry/exit distances along the ray (tmin may be negative if the
// origin is inside the box).
inline bool ray_box(const Ray& ray, const AABB& box, double& tmin_out, double& tmax_out) {
    double tmin = -std::numeric_limits<double>::infinity();
    double tmax =  std::numeric_limits<double>::infinity();

    const double origin[3] = { ray.origin.x, ray.origin.y, ray.origin.z };
    const double dir[3]    = { ray.dir.x,    ray.dir.y,    ray.dir.z };
    const double bmin[3]   = { box.min.x,    box.min.y,    box.min.z };
    const double bmax[3]   = { box.max.x,    box.max.y,    box.max.z };

    for (int axis = 0; axis < 3; ++axis) {
        double inv_d = 1.0 / dir[axis];
        double t1 = (bmin[axis] - origin[axis]) * inv_d;
        double t2 = (bmax[axis] - origin[axis]) * inv_d;
        if (inv_d < 0.0) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        if (tmax < tmin) return false;
    }

    tmin_out = tmin;
    tmax_out = tmax;
    return tmax >= std::max(tmin, 0.0);
}

// Union of two boxes — the operation every internal node's box is
// built from.
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

// --- Tree node ----------------------------------------------------------
// Leaf iff children are both null. Deliberately no separate bool tag —
// null children is the leaf condition, one less thing to keep in sync.
struct BVHNode {
    AABB box;
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;
    std::vector<const ScatObject*> objects;   // populated only at leaves
};

// --- BVH ------------------------------------------------------------------
class BVH {
public:
    // Takes non-owning pointers — the caller keeps its ScatObjects alive
    // for the tree's lifetime.
    explicit BVH(std::vector<const ScatObject*> objects, int leaf_threshold = 1)
        : leaf_threshold_(leaf_threshold)
    {
        root_ = build(std::move(objects));
    }

    // Closest hit anywhere in the tree, or nullopt.
    std::optional<Hit> intersect(const Ray& ray) const {
        return traverse(root_.get(), ray);
    }

private:
    std::unique_ptr<BVHNode> root_;
    int leaf_threshold_;

    std::unique_ptr<BVHNode> build(std::vector<const ScatObject*> objects) {
        auto node = std::make_unique<BVHNode>();

        // Node's box: union of every object's box in this group.
        node->box = objects.front()->bounding_box();
        for (size_t i = 1; i < objects.size(); ++i)
            node->box = union_box(node->box, objects[i]->bounding_box());

        if (static_cast<int>(objects.size()) <= leaf_threshold_) {
            node->objects = std::move(objects);
            return node;   // left/right stay null -> this is a leaf
        }

        // Axis with the largest centroid spread among THIS group.
        Vec3 c_min = objects.front()->centroid();
        Vec3 c_max = c_min;
        for (auto* obj : objects) {
            const Vec3& c = obj->centroid();
            c_min.x = std::min(c_min.x, c.x); c_max.x = std::max(c_max.x, c.x);
            c_min.y = std::min(c_min.y, c.y); c_max.y = std::max(c_max.y, c.y);
            c_min.z = std::min(c_min.z, c.z); c_max.z = std::max(c_max.z, c.z);
        }
        double spread[3] = { c_max.x - c_min.x, c_max.y - c_min.y, c_max.z - c_min.z };
        int axis = std::distance(spread, std::max_element(spread, spread + 3));

        // Median split by centroid along that axis.
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

        node->left  = build(std::move(left_objs));
        node->right = build(std::move(right_objs));
        return node;
    }

    std::optional<Hit> traverse(const BVHNode* node, const Ray& ray) const {
        double tmin, tmax;
        if (!ray_box(ray, node->box, tmin, tmax))
            return std::nullopt;

        // Leaf: test every object directly, keep the closest.
        if (!node->left && !node->right) {
            std::optional<Hit> best;
            for (auto* obj : node->objects) {
                auto hit = obj->intersect(ray);
                if (hit && (!best || hit->t < best->t))
                    best = hit;
            }
            return best;
        }

        // Internal: try both children, keep whichever gives the closer hit.
        // (No near/far ordering or best-so-far pruning yet — correctness
        // first, per the plan; this is the first thing to optimize later.)
        auto hit_left  = node->left  ? traverse(node->left.get(),  ray) : std::nullopt;
        auto hit_right = node->right ? traverse(node->right.get(), ray) : std::nullopt;

        if (hit_left && hit_right)
            return (hit_left->t < hit_right->t) ? hit_left : hit_right;
        return hit_left ? hit_left : hit_right;
    }
};