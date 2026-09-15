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
#include <unordered_map>
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
//     matches the geometry;
//   - the self-intersection nudge after a bounce is sized off the largest
//     curvature padding any patch's stage-1 test uses (SurfacePatch::
//     max_sagitta()), not an arbitrary tiny constant — that padding tests
//     triangles pushed off the true surface by up to +-sagitta, so a nudge
//     smaller than that can leave the next ray still inside the padded
//     shell, re-detecting the same point and getting stuck bouncing in place.
//
// trace_ray() and self_intersect_epsilon() are public (not just used
// internally by run()) so other drivers built on top of this one — e.g.
// BounceMapDriver, which generates its own ray origins directly on a
// patch's surface rather than going through set_rays() — can reuse the same
// tracing and self-intersection-safe launch logic instead of duplicating it.
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

        // Which patch (as a ScatObject*) each recorded bounce hit, in the
        // same order as bounces happen — hit_objects[k] is the patch that
        // produced points[k+1] (points[0] is always the source, which isn't
        // a hit on anything). NOT extended with an entry for an escaping
        // ray's synthetic tail point. Populated by trace_ray() using the
        // Hit::object BVH::intersect() tags each hit with; callers that
        // need per-patch hit counts (e.g. PlaneSourceDriver) read this
        // instead of re-deriving identity some other way.
        std::vector<const ScatObject*> hit_objects;
    };

    // Loads every "*.txt" regular file in `patch_directory` as a
    // SurfacePatch and builds one BVH over all of them. Throws if the
    // directory doesn't exist, has no patches, or a patch file is malformed.
    explicit RayTracingDriver(const std::string& patch_directory) {
        load_patches(patch_directory);
        build_bvh();
        compute_scene_scale();
    }

    // Loads exactly the files "<directory><file_prefix><i>.txt" for i in
    // [first_index_1based, last_index_1based] inclusive — 1-based, matching
    // the on-disk naming IFGF-RP itself uses (`directory` should include
    // its own trailing slash). Unlike the directory-scanning constructor
    // above, this builds paths directly rather than scanning + sorting: a
    // lexicographic sort of filenames does NOT match numeric order once
    // indices hit two digits (e.g. "-10.txt" sorts before "-2.txt"), which
    // would silently scramble which index means which file. Throws if the
    // range is empty/invalid or any file in it is missing/malformed (same
    // fail-loud behavior as the other constructor).
    RayTracingDriver(const std::string& directory,
                      const std::string& file_prefix,
                      int first_index_1based,
                      int last_index_1based) {
        load_patches_range(directory, file_prefix, first_index_1based, last_index_1based);
        build_bvh();
        compute_scene_scale();
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
            paths_.push_back(trace_ray(sources_[i], directions_[i], max_bounces));
        }
    }

    const std::vector<RayPath>& paths() const { return paths_; }
    const std::vector<std::unique_ptr<SurfacePatch>>& patches() const { return patches_; }

    // Maps a hit's Hit::object (as recorded in RayPath::hit_objects) back to
    // its index into patches(). Returns -1 for a pointer this driver didn't
    // load (shouldn't happen for a Hit produced by this driver's own BVH,
    // but checked rather than assumed). Built once, right after loading.
    long long patch_index_of(const ScatObject* object) const {
        auto it = patch_index_by_ptr_.find(object);
        return it != patch_index_by_ptr_.end() ? static_cast<long long>(it->second) : -1;
    }

    // Traces a single ray from an explicit (origin, direction): bounces
    // specularly off whatever it hits until it escapes the scene, hits
    // `max_bounces` without escaping (Trapped), or a patch intersection
    // throws (Error, recorded on the returned path rather than propagated —
    // this method never throws for that reason). `direction` need not be
    // pre-normalized; a zero-length one comes back as an Error path rather
    // than throwing, since (unlike set_rays()) this is a per-call, not a
    // batch-setup, entry point.
    RayPath trace_ray(const Vec3& origin, const Vec3& direction, int max_bounces) const {
        RayPath path;
        path.points.push_back(origin);

        const double len2 = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
        if (len2 < 1e-300) {
            path.status = RayPath::Status::Error;
            path.error_message = "RayTracingDriver::trace_ray: zero-length direction";
            return path;
        }

        Vec3 origin_pt = origin;
        Vec3 dir = normalize_dir(direction);

        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            Ray ray{ origin_pt, dir };
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
                path.points.push_back({ origin_pt.x + dir.x * scene_diagonal_,
                                         origin_pt.y + dir.y * scene_diagonal_,
                                         origin_pt.z + dir.z * scene_diagonal_ });
                return path;
            }

            path.points.push_back(hit->point);
            path.hit_objects.push_back(hit->object);
            path.bounce_count++;

            dir = reflect(dir, hit->normal);

            const double eps = self_intersect_epsilon(hit->object);
            origin_pt = { hit->point.x + dir.x * eps,
                          hit->point.y + dir.y * eps,
                          hit->point.z + dir.z * eps };
        }

        path.status = RayPath::Status::Trapped;
        return path;
    }

    // The minimum safe distance to nudge a new ray's origin away from a
    // surface point it's leaving (after a bounce, or — for a caller
    // launching rays directly from a patch's surface, like BounceMapDriver —
    // at the very start) so the next intersect() call can't re-detect that
    // same point through that object's curvature-padded stage-1 test.
    //
    // `leaving_object` should be the specific patch the ray is leaving (a
    // bounce's hit->object, or the patch a caller is launching straight off
    // of) — its own self_intersect_padding() is what matters, NOT the
    // worst-case padding anywhere else in the scene. Pass nullptr only when
    // no specific object applies (e.g. a source with no proximate surface);
    // that falls back to the scene-wide max_patch_sagitta_, which is
    // conservative but can over-nudge a low-curvature patch sitting in a
    // scene that also contains a high-curvature one — see
    // ScatObject::self_intersect_padding()'s comment for why a scene-wide
    // value isn't used as the general case.
    double self_intersect_epsilon(const ScatObject* leaving_object = nullptr) const {
        const double padding = leaving_object ? leaving_object->self_intersect_padding()
                                               : max_patch_sagitta_;
        return std::max(kSelfIntersectEpsilonFactor * scene_diagonal_,
                         kSagittaClearanceFactor * padding);
    }

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
    double max_patch_sagitta_ = 0.0;
    std::unordered_map<const ScatObject*, size_t> patch_index_by_ptr_;

    // Fraction of the scene diagonal used as a floor under the self-
    // intersection nudge, for the degenerate case where every patch is
    // perfectly flat (max_patch_sagitta_ == 0, so that alone would nudge by
    // nothing at all).
    static constexpr double kSelfIntersectEpsilonFactor = 1e-9;

    // Safety multiple on max_patch_sagitta_ for that nudge: SurfacePatch's
    // stage-1 test checks triangles pushed up to +-sagitta off the true
    // surface (see SurfacePatch::compute_cell_padding()), so a reflected
    // ray has to clear a good deal more than sagitta itself to be sure it's
    // outside every padded copy — otherwise the very next intersect() call
    // can re-detect the point it just left through that padding, and the
    // ray gets stuck bouncing in place instead of moving on.
    static constexpr double kSagittaClearanceFactor = 5.0;

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
        patch_index_by_ptr_.reserve(patches_.size());
        for (size_t i = 0; i < patches_.size(); ++i) {
            objects.push_back(patches_[i].get());
            patch_index_by_ptr_[patches_[i].get()] = i;
        }
        bvh_ = std::make_unique<BVH>(std::move(objects));
    }

    void compute_scene_scale() {
        AABB box = patches_.front()->bounding_box();
        for (const auto& patch : patches_) {
            box = union_box(box, patch->bounding_box());
            max_patch_sagitta_ = std::max(max_patch_sagitta_, patch->max_sagitta());
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

    void load_patches_range(const std::string& directory, const std::string& file_prefix,
                             int first_index_1based, int last_index_1based) {
        if (last_index_1based < first_index_1based) {
            throw std::invalid_argument(
                "RayTracingDriver: empty patch index range [" + std::to_string(first_index_1based) +
                ", " + std::to_string(last_index_1based) + "]");
        }

        patches_.reserve(static_cast<size_t>(last_index_1based - first_index_1based + 1));
        for (int i = first_index_1based; i <= last_index_1based; ++i) {
            // SurfacePatch's constructor throws on a missing/malformed file
            // — that propagates straight out of here, failing the whole load.
            patches_.push_back(std::make_unique<SurfacePatch>(directory + file_prefix + std::to_string(i) + ".txt"));
        }
    }
};
