#pragma once

#include <optional>
#include <limits>

// --- Minimal supporting types -------------------------------------------
// Replace with your existing Vec3/Ray types if you already have them
// elsewhere in the codebase — nothing below depends on this exact definition.

struct Vec3 {
    double x, y, z;
};

struct Ray {
    Vec3 origin;
    Vec3 dir;
};

struct AABB {
    Vec3 min{ std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity() };
    Vec3 max{ -std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity() };
};

// What an intersection reports back, regardless of what kind of
// ScatObject was hit.
struct Hit {
    double t;        // distance along the ray
    Vec3 point;      // 3D hit location
    Vec3 normal;      // surface normal at the hit, unit length
};

// --- ScatObject interface -----------------------------------------------------
// Anything the BVH can hold: flat facets, curved patches, primitives
// used for testing (spheres, triangles), etc. The tree/traversal code
// only ever touches this interface — it never needs to know the
// concrete type behind it.

class ScatObject {
public:
    virtual ~ScatObject() = default;

    // Cached, computed once at construction time by the derived type.
    // Plain data — not virtual, never recomputed after construction.
    const AABB&  bounding_box() const { return box_; }
    const Vec3&  centroid()     const { return centroid_; }

    // The one real piece of polymorphism: does this ray hit me, and
    // where. Returns nullopt on a miss. Concrete types implement this
    // however is appropriate (closed-form for a flat facet, sample-mesh
    // + Newton-polish for a curved patch, etc).
    virtual std::optional<Hit> intersect(const Ray& ray) const = 0;

protected:
    // Derived constructors are responsible for filling these in from
    // whatever geometry they hold, before the ScatObject is considered
    // usable (e.g. before it's handed to the BVH builder).
    AABB box_;
    Vec3 centroid_;
};