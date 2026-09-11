#pragma once

#include "BVH.hpp"
#include "ScatObject.hpp"
#include "ScatObjects/SurfacePatch.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// --- RayTracingDriver --------------------------------------------------------
// Orchestrates the whole "billiard ball" ray tracing pipeline over a
// directory of IFGF-RP surface patches:
//   1. load every "*.txt" patch file in a directory into a SurfacePatch,
//      build one BVH over all of them;
//   2. take a set of source points and paired ray directions;
//   3. for each ray, bounce it specularly off whatever it hits until it
//      escapes the scene, hits a bounce cap, or a patch intersection throws;
//   4. optionally dump every traced path to a legacy VTK PolyData file as
//      line segments, for visualization (e.g. in ParaView).
//
// Design decisions (settled in conversation, not re-litigated here):
//   - sources/directions are paired 1:1 — ray i is sources[i] -> directions[i],
//     the two lists must be the same length (set_rays() throws otherwise);
//   - a ray that throws mid-trace (SurfacePatch::intersect's Newton
//     refinement can throw on non-convergence, a singular Jacobian, or an
//     out-of-domain result — see SurfacePatch.hpp) fails only that ray: the
//     error is recorded on its RayPath and tracing continues with the rest;
//   - only "*.txt" files in the directory are loaded as patches; a malformed
//     one still throws (fail loud, whole load aborts) — non-.txt files are
//     silently skipped;
//   - an escaping ray's final segment is drawn one scene-bounding-box
//     diagonal past its last point, so the tail is visible at a scale that
//     matches the geometry.
//
// Like SurfacePatch.hpp, this pulls in MKL/MPI/OpenMP transitively via
// Utils/PatchInterpolation.hpp — see SurfacePatch.hpp's header comment.
class RayTracingDriver {
public:
    // One traced ray's outcome.
    struct RayPath {
        enum class Status { Escaped, Trapped, Error };

        std::vector<Vec3> points;   // source, then each successive hit point,
                                     // then (if Escaped) one synthetic tail point
        Status status = Status::Trapped;
        int bounce_count = 0;       // number of surface hits actually recorded
        std::string error_message;  // populated only when status == Error
    };

    // Loads every "*.txt" regular file in `patch_directory` as a
    // SurfacePatch and builds one BVH over all of them. Throws if the
    // directory doesn't exist, has no patches, or a patch file is malformed.
    explicit RayTracingDriver(const std::string& patch_directory) {
        load_patches(patch_directory);
        build_bvh();
        compute_scene_diagonal();
    }

    // Configures the rays to trace. Paired 1:1: ray i starts at sources[i]
    // and travels along directions[i] — the two lists must be the same
    // length, and no direction may be zero-length.
    void set_rays(std::vector<Vec3> sources, std::vector<Vec3> directions) {
        if (sources.size() != directions.size()) {
            throw std::invalid_argument(
                "RayTracingDriver::set_rays: sources and directions must be the same "
                "length (paired 1:1) — got " + std::to_string(sources.size()) +
                " sources and " + std::to_string(directions.size()) + " directions");
        }
        for (size_t i = 0; i < directions.size(); ++i) {
            const Vec3& d = directions[i];
            double len2 = d.x * d.x + d.y * d.y + d.z * d.z;
            if (len2 < 1e-300) {
                throw std::invalid_argument(
                    "RayTracingDriver::set_rays: direction " + std::to_string(i) + " has zero length");
            }
        }
        sources_ = std::move(sources);
        directions_ = std::move(directions);
    }

    // Traces every configured ray: bounces specularly off whatever it hits
    // (reflecting about the hit normal) until it escapes the scene (BVH
    // reports no hit), hits `max_bounces` without escaping (Trapped), or a
    // patch intersection throws (Error — that ray only; tracing continues
    // with the rest).
    void run(int max_bounces = 50) {
        paths_.clear();
        paths_.reserve(sources_.size());

        for (size_t i = 0; i < sources_.size(); ++i) {
            paths_.push_back(trace_one_ray(sources_[i], directions_[i], max_bounces));
        }
    }

    const std::vector<RayPath>& paths() const { return paths_; }
    const std::vector<std::unique_ptr<SurfacePatch>>& patches() const { return patches_; }

    // Writes every traced path as legacy ASCII VTK PolyData: one 2-point
    // LINE cell per bounce leg, with per-cell RayID/BounceIndex scalars so
    // individual rays or bounces can be picked out or colored in ParaView.
    void write_vtk(const std::string& filename) const {
        std::vector<Vec3> all_points;
        struct Segment { int p0, p1, ray_id, bounce_index; };
        std::vector<Segment> segments;

        for (size_t ray_id = 0; ray_id < paths_.size(); ++ray_id) {
            const RayPath& path = paths_[ray_id];
            const int base = static_cast<int>(all_points.size());
            for (const Vec3& p : path.points) all_points.push_back(p);

            for (size_t k = 0; k + 1 < path.points.size(); ++k) {
                segments.push_back({ base + static_cast<int>(k), base + static_cast<int>(k) + 1,
                                      static_cast<int>(ray_id), static_cast<int>(k) });
            }
        }

        std::ofstream out(filename);
        if (!out.is_open()) {
            throw std::runtime_error("RayTracingDriver::write_vtk: cannot open output file: " + filename);
        }

        out << "# vtk DataFile Version 3.0\n";
        out << "Billiard-ball ray traces\n";
        out << "ASCII\n";
        out << "DATASET POLYDATA\n";

        out << "POINTS " << all_points.size() << " double\n";
        for (const Vec3& p : all_points) {
            out << p.x << ' ' << p.y << ' ' << p.z << '\n';
        }

        out << "LINES " << segments.size() << ' ' << segments.size() * 3 << '\n';
        for (const Segment& s : segments) {
            out << "2 " << s.p0 << ' ' << s.p1 << '\n';
        }

        out << "CELL_DATA " << segments.size() << '\n';
        out << "SCALARS RayID int 1\nLOOKUP_TABLE default\n";
        for (const Segment& s : segments) out << s.ray_id << '\n';
        out << "SCALARS BounceIndex int 1\nLOOKUP_TABLE default\n";
        for (const Segment& s : segments) out << s.bounce_index << '\n';
    }

private:
    std::vector<std::unique_ptr<SurfacePatch>> patches_;
    std::unique_ptr<BVH> bvh_;
    double scene_diagonal_ = 0.0;

    std::vector<Vec3> sources_;
    std::vector<Vec3> directions_;
    std::vector<RayPath> paths_;

    // Fraction of the scene diagonal used to nudge a reflected ray's origin
    // off the surface it just left, so it doesn't immediately re-hit the
    // same point due to floating-point roundoff.
    static constexpr double kSelfIntersectEpsilonFactor = 1e-9;

    void load_patches(const std::string& patch_directory) {
        namespace fs = std::filesystem;

        if (!fs::exists(patch_directory) || !fs::is_directory(patch_directory)) {
            throw std::runtime_error("RayTracingDriver: not a directory: " + patch_directory);
        }

        std::vector<fs::path> txt_files;
        for (const auto& entry : fs::directory_iterator(patch_directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                txt_files.push_back(entry.path());
            }
        }

        // Deterministic order — directory_iterator's order isn't guaranteed.
        std::sort(txt_files.begin(), txt_files.end());

        patches_.reserve(txt_files.size());
        for (const auto& path : txt_files) {
            // SurfacePatch's constructor throws on a malformed file — that
            // propagates straight out of here, failing the whole load.
            patches_.push_back(std::make_unique<SurfacePatch>(path.string()));
        }

        if (patches_.empty()) {
            throw std::runtime_error("RayTracingDriver: no *.txt patch files found in: " + patch_directory);
        }
    }

    void build_bvh() {
        std::vector<const ScatObject*> objects;
        objects.reserve(patches_.size());
        for (const auto& patch : patches_) objects.push_back(patch.get());
        bvh_ = std::make_unique<BVH>(std::move(objects));
    }

    void compute_scene_diagonal() {
        AABB box = patches_.front()->bounding_box();
        for (const auto& patch : patches_) {
            box = union_box(box, patch->bounding_box());
        }
        const double dx = box.max.x - box.min.x;
        const double dy = box.max.y - box.min.y;
        const double dz = box.max.z - box.min.z;
        scene_diagonal_ = std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    static Vec3 normalize_dir(const Vec3& v) {
        const double len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        return { v.x / len, v.y / len, v.z / len };
    }

    // Specular reflection of direction d about unit normal n. Sign of n
    // doesn't matter: flipping n leaves the result unchanged.
    static Vec3 reflect(const Vec3& d, const Vec3& n) {
        const double dn = d.x * n.x + d.y * n.y + d.z * n.z;
        return { d.x - 2.0 * dn * n.x, d.y - 2.0 * dn * n.y, d.z - 2.0 * dn * n.z };
    }

    RayPath trace_one_ray(const Vec3& source, const Vec3& direction, int max_bounces) const {
        RayPath path;
        path.points.push_back(source);

        Vec3 origin = source;
        Vec3 dir = normalize_dir(direction);   // non-zero guaranteed by set_rays()

        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            Ray ray{ origin, dir };
            std::optional<Hit> hit;

            try {
                hit = bvh_->intersect(ray);
            } catch (const std::exception& e) {
                path.status = RayPath::Status::Error;
                path.error_message = e.what();
                return path;
            }

            if (!hit) {
                path.status = RayPath::Status::Escaped;
                path.points.push_back({ origin.x + dir.x * scene_diagonal_,
                                         origin.y + dir.y * scene_diagonal_,
                                         origin.z + dir.z * scene_diagonal_ });
                return path;
            }

            path.points.push_back(hit->point);
            path.bounce_count++;

            dir = reflect(dir, hit->normal);

            const double eps = kSelfIntersectEpsilonFactor * scene_diagonal_;
            origin = { hit->point.x + dir.x * eps,
                       hit->point.y + dir.y * eps,
                       hit->point.z + dir.z * eps };
        }

        path.status = RayPath::Status::Trapped;
        return path;
    }
};
