#pragma once

#include "RayTracingDriver.hpp"
#include "ScatObjects/SurfacePatch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// --- PlaneSourceDriver -------------------------------------------------------
// Illuminates a whole patch directory from a finite, axis-aligned rectangular
// plane of source points, all launched along one fixed direction, and counts
// how many times each patch gets hit by MULTIPLE-SCATTERING bounces — each
// ray's direct hit straight off the source plane doesn't count by default
// (see the `count_direct_hit` design decision below). Writes the result as
// one VTK surface per patch (at each patch's own native node-grid
// resolution) with a constant "HitCount" scalar over that patch.
//
// Design decisions (settled in conversation, not re-litigated here):
//   - the plane is defined by two corner points and is axis-aligned: exactly
//     one of x/y/z must match between the two corners (within a small
//     relative tolerance), which is taken as the plane's normal axis — the
//     other two coordinates span the rectangle. This only covers a plane
//     perpendicular to a coordinate axis, not an arbitrary tilt, in exchange
//     for needing nothing more than two points to describe it;
//   - the plane's tensor-product source grid uses independent nx x ny point
//     counts (not a single shared n) so a non-square rectangle keeps even
//     spacing in both directions; grid points run corner-to-corner inclusive
//     (this is a physical sampling grid of launch points, not a solid-angle
//     sampling, so there's no reason to inset from the edges);
//   - every ray shares the same fixed launch direction (unlike
//     RayTracingDriver's paired sources/directions);
//   - a patch's hit count increments once per (counted) bounce that lands on
//     it, not once per ray — a ray that bounces three times off the same
//     patch adds 3, not 1;
//   - each ray's FIRST hit (direct illumination straight from the source
//     plane, with no prior bounce) is excluded from the count by default —
//     this driver exists to map multiple-scattering activity, so a patch lit
//     only directly reads 0 unless `run()`'s `count_direct_hit` is set;
//   - the whole directory's patches are loaded and ALL of them are written
//     out (hit count 0 where nothing landed), so the output shows
//     illuminated and shadowed geometry together in one file;
//   - the output surface uses each patch's own raw node grid (imax x jmax,
//     the actual IFGF-RP points — see SurfacePatch::data()), not a
//     resampled grid, since this is meant to show the original patches.
//
// Reuses RayTracingDriver::trace_ray() for the entire bounce/error/reflection
// pipeline; the only new piece of plumbing this needed was Hit::object (see
// ScatObject.hpp) and RayPath::hit_objects / patch_index_of() (see
// RayTracingDriver.hpp) to recover which patch each bounce landed on.
class PlaneSourceDriver {
public:
    // Loads every "*.txt" patch in `patch_directory` (see RayTracingDriver's
    // directory constructor) and builds one BVH over all of them.
    explicit PlaneSourceDriver(const std::string& patch_directory)
        : driver_(patch_directory)
    {}

    // Traces one ray per point of an nx x ny grid spanning the axis-aligned
    // rectangle with opposite corners `corner0`/`corner1`, all launched
    // along the same `direction`, against the whole scene. `corner0` and
    // `corner1` must agree (within a small relative tolerance) in exactly
    // one coordinate — that axis is taken as the plane's normal, and the
    // other two are spanned by nx and ny respectively, in x/y/z order (e.g.
    // a plane constant in y varies x then z, with nx samples of x and ny of
    // z). Every recorded bounce (see RayTracingDriver::trace_ray) increments
    // that patch's hit count, EXCEPT each ray's very first hit — the direct
    // illumination straight off the source plane — unless `count_direct_hit`
    // is set: this driver is meant to map multiple-scattering activity, so
    // a patch lit only by direct line-of-sight from the source (never by a
    // ray already bounced at least once) reads 0 by default. A ray whose
    // trace errors mid-flight still counts whatever (non-direct) bounces it
    // recorded before the error. Parallelized over the nx*ny source points.
    // Safe to call again; results replace any prior run.
    void run(const Vec3& corner0, const Vec3& corner1, const Vec3& direction,
             int nx, int ny, int max_bounces = 50, bool count_direct_hit = false) {
        if (nx < 1 || ny < 1) {
            throw std::invalid_argument("PlaneSourceDriver::run: nx and ny must both be >= 1");
        }
        const double dir_len2 = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
        if (dir_len2 < 1e-300) {
            throw std::invalid_argument("PlaneSourceDriver::run: direction has zero length");
        }

        const std::vector<Vec3> sources = build_plane_grid(corner0, corner1, nx, ny);

        const auto& patches = driver_.patches();
        hit_count_.assign(patches.size(), 0);

        const long long n_sources = static_cast<long long>(sources.size());

        #pragma omp parallel for schedule(dynamic)
        for (long long k = 0; k < n_sources; ++k) {
            const RayTracingDriver::RayPath path =
                driver_.trace_ray(sources[static_cast<size_t>(k)], direction, max_bounces);

            const size_t first_counted = count_direct_hit ? 0 : 1;  // skip index 0: the direct hit
            for (size_t hit_idx = first_counted; hit_idx < path.hit_objects.size(); ++hit_idx) {
                const long long idx = driver_.patch_index_of(path.hit_objects[hit_idx]);
                if (idx < 0) continue;  // defensive: shouldn't happen for our own BVH's hits
                #pragma omp atomic
                hit_count_[static_cast<size_t>(idx)]++;
            }
        }
    }

    // Writes every patch in the directory as its own triangulated legacy
    // ASCII VTK PolyData surface, using that patch's native imax x jmax node
    // grid, with a constant "HitCount" point scalar equal to the number of
    // bounces run() recorded against it. Call run() first.
    void write_vtk(const std::string& filename) const {
        if (hit_count_.empty()) {
            throw std::runtime_error("PlaneSourceDriver::write_vtk: no data — call run() first");
        }

        const auto& patches = driver_.patches();

        std::vector<Vec3> all_points;
        std::vector<long long> all_hits;
        struct Tri { int a, b, c; };
        std::vector<Tri> tris;

        for (size_t p = 0; p < patches.size(); ++p) {
            const SurfacePatch::Data& d = patches[p]->data();
            const int imax = d.imax;
            const int jmax = d.jmax;
            const int base = static_cast<int>(all_points.size());
            const long long hits = hit_count_[p];

            for (int idx = 0; idx < imax * jmax; ++idx) {
                all_points.push_back({ d.x[idx], d.y[idx], d.z[idx] });
                all_hits.push_back(hits);
            }

            // Same diagonal split convention as SurfacePatch's own coarse
            // grid / BounceMapDriver's output: (p00,p10,p11) and
            // (p00,p11,p01) per cell, flat index i*jmax + j (matching the
            // file's own row-major layout).
            for (int i = 0; i + 1 < imax; ++i) {
                for (int j = 0; j + 1 < jmax; ++j) {
                    const int p00 = base + (i * jmax + j);
                    const int p10 = base + ((i + 1) * jmax + j);
                    const int p01 = base + (i * jmax + (j + 1));
                    const int p11 = base + ((i + 1) * jmax + (j + 1));
                    tris.push_back({ p00, p10, p11 });
                    tris.push_back({ p00, p11, p01 });
                }
            }
        }

        std::ofstream out(filename);
        if (!out.is_open()) {
            throw std::runtime_error("PlaneSourceDriver::write_vtk: cannot open output file: " + filename);
        }

        out << "# vtk DataFile Version 3.0\n";
        out << "Plane-source patch hit counts\n";
        out << "ASCII\n";
        out << "DATASET POLYDATA\n";

        out << "POINTS " << all_points.size() << " double\n";
        for (const Vec3& pt : all_points) {
            out << pt.x << ' ' << pt.y << ' ' << pt.z << '\n';
        }

        out << "POLYGONS " << tris.size() << ' ' << tris.size() * 4 << '\n';
        for (const Tri& t : tris) {
            out << "3 " << t.a << ' ' << t.b << ' ' << t.c << '\n';
        }

        out << "POINT_DATA " << all_points.size() << '\n';
        out << "SCALARS HitCount long 1\nLOOKUP_TABLE default\n";
        for (long long h : all_hits) out << h << '\n';
    }

    const RayTracingDriver& driver() const { return driver_; }

private:
    RayTracingDriver driver_;
    std::vector<long long> hit_count_;  // size == driver_.patches().size(), indexed by patch_index_of()

    // Builds the nx*ny axis-aligned source grid between corner0 and
    // corner1, row-major over (axis_a index, axis_b index) where axis_a/b
    // are whichever two of x/y/z aren't the (near-)constant one. Throws if
    // the two corners don't agree in exactly one coordinate.
    static std::vector<Vec3> build_plane_grid(const Vec3& corner0, const Vec3& corner1, int nx, int ny) {
        const double c0[3] = { corner0.x, corner0.y, corner0.z };
        const double c1[3] = { corner1.x, corner1.y, corner1.z };

        const double scale = std::max({ 1.0, std::fabs(c0[0]), std::fabs(c0[1]), std::fabs(c0[2]),
                                              std::fabs(c1[0]), std::fabs(c1[1]), std::fabs(c1[2]) });
        const double tol = 1e-9 * scale;

        int constant_axis = -1;
        int match_count = 0;
        for (int a = 0; a < 3; ++a) {
            if (std::fabs(c0[a] - c1[a]) < tol) {
                constant_axis = a;
                ++match_count;
            }
        }
        if (match_count != 1) {
            throw std::invalid_argument(
                "PlaneSourceDriver: corner0 and corner1 must agree in exactly one coordinate "
                "to define an axis-aligned plane (got " + std::to_string(match_count) +
                " matching coordinates)");
        }

        // The two non-constant axes, in x/y/z order — first spanned by nx,
        // second by ny.
        int axis_a = -1, axis_b = -1;
        for (int a = 0; a < 3; ++a) {
            if (a == constant_axis) continue;
            if (axis_a < 0) axis_a = a; else axis_b = a;
        }

        std::vector<double> grid_a(nx), grid_b(ny);
        for (int i = 0; i < nx; ++i) {
            grid_a[i] = (nx == 1) ? c0[axis_a]
                                  : c0[axis_a] + (c1[axis_a] - c0[axis_a]) * static_cast<double>(i) / (nx - 1);
        }
        for (int j = 0; j < ny; ++j) {
            grid_b[j] = (ny == 1) ? c0[axis_b]
                                  : c0[axis_b] + (c1[axis_b] - c0[axis_b]) * static_cast<double>(j) / (ny - 1);
        }

        std::vector<Vec3> points;
        points.reserve(static_cast<size_t>(nx) * ny);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                double coord[3];
                coord[constant_axis] = c0[constant_axis];
                coord[axis_a] = grid_a[i];
                coord[axis_b] = grid_b[j];
                points.push_back({ coord[0], coord[1], coord[2] });
            }
        }
        return points;
    }
};
