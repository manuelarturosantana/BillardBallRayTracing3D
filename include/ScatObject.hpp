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

// Forward-declared so Hit can point back at the object it came from;
// ScatObject itself is defined just below.
class ScatObject;

// What an intersection reports back, regardless of what kind of
// ScatObject was hit.
struct Hit {
    double t;        // distance along the ray
    Vec3 point;      // 3D hit location
    Vec3 normal;      // surface normal at the hit, unit length

    // Which object this hit came from. Left null by every ScatObject's own
    // intersect() (a Triangle or SurfacePatch has no reason to know its own
    // address); BVH::intersect() fills this in on the winning hit, since it
    // already has the object pointer in hand when it picks the closest one.
    // Callers that need to know which patch a ray landed on (e.g. counting
    // per-patch hits) read this off the Hit rather than threading identity
    // through some other channel.
    const ScatObject* object = nullptr;
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

    // How far a ray leaving this object's surface (a bounce off it, or a
    // launch directly from it) must clear before the next intersect() call
    // is safe from re-detecting the same point through this object's own
    // stage-1 padding. Flat objects (e.g. Triangle) have no such padding, so
    // 0.0 is the right default; SurfacePatch overrides this with its own
    // max_sagitta(). Deliberately per-object rather than a single scene-wide
    // worst case — a scene-wide max means one high-curvature patch anywhere
    // inflates the nudge for every ray in the scene, including ones nowhere
    // near it, which can push a ray far enough off a nearby low-curvature
    // patch's surface that it misses that patch entirely instead of bouncing
    // off it (see RayTracingDriver::self_intersect_epsilon()).
    virtual double self_intersect_padding() const { return 0.0; }

protected:
    // Derived constructors are responsible for filling these in from
    // whatever geometry they hold, before the ScatObject is considered
    // usable (e.g. before it's handed to the BVH builder).
    AABB box_;
    Vec3 centroid_;
};