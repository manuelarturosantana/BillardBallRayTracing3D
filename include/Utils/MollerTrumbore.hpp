#pragma once

#include "../ScatObject.hpp"

#include <cmath>
#include <optional>

// --- Moller-Trumbore ray/triangle intersection ------------------------------
// Pulled out of Triangle.hpp so any ScatObject can test a ray against a flat
// triangle — Triangle uses it directly on its one triangle, SurfacePatch uses
// it per-cell against the two triangles a coarse grid square splits into.

inline Vec3 vec_sub(const Vec3& a, const Vec3& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}
inline Vec3 vec_cross(const Vec3& a, const Vec3& b) {
    return { a.y * b.z - a.z * b.y,
             a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x };
}
inline double vec_dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline Vec3 vec_normalize(const Vec3& v) {
    double len = std::sqrt(vec_dot(v, v));
    return { v.x / len, v.y / len, v.z / len };
}

// Ray vs. triangle (v0, v1, v2). Returns nullopt on a miss: ray parallel to
// the triangle's plane, barycentric coords outside the triangle, or the hit
// behind the ray origin. `eps` guards the parallel test.
inline std::optional<Hit> moller_trumbore(const Ray& ray, const Vec3& v0, const Vec3& v1, const Vec3& v2,
                                           double eps = 1e-9) {
    Vec3 edge1 = vec_sub(v1, v0);
    Vec3 edge2 = vec_sub(v2, v0);
    Vec3 pvec  = vec_cross(ray.dir, edge2);
    double det = vec_dot(edge1, pvec);

    if (std::fabs(det) < eps)
        return std::nullopt;   // ray parallel to triangle plane

    double inv_det = 1.0 / det;
    Vec3 tvec = vec_sub(ray.origin, v0);
    double u = vec_dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
        return std::nullopt;

    Vec3 qvec = vec_cross(tvec, edge1);
    double v = vec_dot(ray.dir, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0)
        return std::nullopt;

    double t = vec_dot(edge2, qvec) * inv_det;
    if (t < eps)
        return std::nullopt;   // behind the ray origin (or self-intersection)

    Hit hit;
    hit.t = t;
    hit.point = { ray.origin.x + t * ray.dir.x,
                  ray.origin.y + t * ray.dir.y,
                  ray.origin.z + t * ray.dir.z };
    hit.normal = vec_normalize(vec_cross(edge1, edge2));
    return hit;
}
