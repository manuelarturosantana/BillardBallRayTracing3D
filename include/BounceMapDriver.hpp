#pragma once

#include "RayTracingDriver.hpp"
#include "ScatObjects/SurfacePatch.hpp"
#include "Utils/MollerTrumbore.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// --- BounceMapDriver ---------------------------------------------------------
// Samples an n x n position grid on a chosen 1-based inclusive range of
// IFGF-RP surface patches and, at every grid point, sweeps a hemisphere of
// outgoing directions about that point's local normal, tracing a
// billiard-ball ray (via RayTracingDriver::trace_ray) for each direction and
// keeping the MAX bounce count seen at that point. The sampled range is
// written out as one combined triangulated VTK surface with per-point
// "MaxBounces" scalars.
//
// Design decisions (settled in conversation, not re-litigated here):
//   - the underlying RayTracingDriver loads EVERY "*.txt" patch in the
//     directory (not just the sampled range) and builds one BVH over all of
//     them, so a ray launched from a sampled patch can bounce off any patch
//     in the scene, not only the ones being sampled;
//   - separately, the patches to actually SAMPLE (launch rays from) are
//     given as a 1-based inclusive index range matching on-disk IFGF-RP
//     naming, loaded by direct path construction rather than picked out of
//     the directory-scanned list — the scan's lexicographic file order does
//     not match numeric index order past 9 patches, so there is no reliable
//     way to recover "patch i" by position in that list; loading it again
//     directly sidesteps that entirely (at the cost of reading that file
//     twice — once for the scene BVH, once for sampling — which is cheap
//     next to the ray tracing itself);
//   - hemisphere directions are sampled solid-angle-uniform (cosine-weighted
//     in theta), not angle-uniform in theta: cell-centered in cos(theta) and
//     phi so each grid cell represents an equal-area bin of the hemisphere,
//     and neither the pole (theta == 0) nor the numerically-grazing equator
//     (theta == pi/2) is ever sampled exactly;
//   - a direction whose trace comes back as an error (SurfacePatch::intersect
//     can throw on Newton non-convergence, a singular Jacobian, or an
//     out-of-domain result deep in the BVH) is skipped — it simply doesn't
//     contribute a bounce count for that point — rather than aborting the
//     whole point or the whole run;
//   - output is one combined triangulated surface, not a point cloud: each
//     sampled patch's n x n grid is triangulated the same way SurfacePatch's
//     own coarse grid is (two triangles per cell), all sampled patches
//     concatenated into one point/polygon list, with MaxBounces as
//     POINT_DATA so it renders as a shaded field directly in ParaView.
//
// Reuses RayTracingDriver::trace_ray() and self_intersect_epsilon() (public
// specifically for this kind of reuse — see RayTracingDriver.hpp) instead of
// duplicating the bounce loop or nudge-sizing logic.
class BounceMapDriver {
public:
    // `patch_directory` is scanned in full (every "*.txt" file becomes part
    // of the traced scene, via RayTracingDriver's directory constructor).
    // Independently, the patches actually sampled are
    // "<patch_directory><file_prefix><i>.txt" for i in
    // [first_index_1based, last_index_1based] inclusive — `patch_directory`
    // should include its own trailing slash, matching RayTracingDriver's
    // range constructor convention.
    BounceMapDriver(const std::string& patch_directory,
                     const std::string& file_prefix,
                     int first_index_1based,
                     int last_index_1based)
        : driver_(patch_directory)
    {
        load_sample_patches(patch_directory, file_prefix, first_index_1based, last_index_1based);
    }

    // For every sampled patch: samples a position_grid_n x position_grid_n
    // grid of surface points (SurfacePatch::sample_grid), and at each point
    // sweeps a theta_n x phi_n solid-angle-uniform hemisphere of outgoing
    // directions about the local normal, tracing each with
    // RayTracingDriver::trace_ray(..., max_bounces) against the FULL scene
    // and keeping the largest bounce_count seen there (directions whose
    // trace errors are skipped). Safe to call again with different
    // resolutions; results replace any prior run.
    void run(int position_grid_n, int theta_n, int phi_n, int max_bounces = 50) {
        if (position_grid_n < 2) {
            throw std::invalid_argument("BounceMapDriver::run: position_grid_n must be >= 2");
        }
        if (theta_n < 1 || phi_n < 1) {
            throw std::invalid_argument("BounceMapDriver::run: theta_n and phi_n must both be >= 1");
        }

        patch_grids_.clear();
        patch_max_bounces_.clear();
        patch_error_counts_.clear();
        patch_grids_.reserve(sample_patches_.size());
        patch_max_bounces_.reserve(sample_patches_.size());
        patch_error_counts_.reserve(sample_patches_.size());

        const size_t num_patches = sample_patches_.size();
        for (size_t patch_num = 0; patch_num < num_patches; ++patch_num) {
            const auto& patch = sample_patches_[patch_num];
            // Sized off THIS patch's own curvature padding, not a scene-wide
            // worst case — see RayTracingDriver::self_intersect_epsilon()'s
            // comment. Using the scene-wide value here would over-nudge (or
            // under-nudge) launches off this patch based on the curvature of
            // some unrelated patch elsewhere in the directory.
            const double eps = driver_.self_intersect_epsilon(patch.get());
            SurfacePatch::SampledGrid grid = patch->sample_grid(position_grid_n);
            const long long nn = static_cast<long long>(grid.x.size());
            std::vector<int> max_bounces_at_point(static_cast<size_t>(nn), 0);
            // Diagnostic: how many of this point's theta_n*phi_n directions
            // came back Error (skipped, uncounted toward `best`) rather than
            // Escaped/Trapped. A point reading MaxBounces == 0 is ambiguous
            // on its own — it could be a ray that genuinely escaped
            // immediately, or one where every direction errored out and
            // `best` never left its initial 0. Written out alongside
            // MaxBounces so the two can be told apart in ParaView.
            std::vector<int> error_count_at_point(static_cast<size_t>(nn), 0);

            // Progress bar for this patch's grid-point sweep: each thread
            // bumps a shared atomic counter as it finishes a point and
            // (throttled, under a critical section so lines don't
            // interleave) redraws the bar in place with \r. The counter is
            // monotonic regardless of which thread's iteration finishes
            // first under dynamic scheduling, so the percentage shown never
            // goes backwards even though completion order is unordered.
            std::atomic<long long> completed{ 0 };
            const long long print_every = std::max<long long>(1, nn / 100);

            #pragma omp parallel for schedule(dynamic)
            for (long long k = 0; k < nn; ++k) {
                const size_t idx = static_cast<size_t>(k);
                const Vec3 point = { grid.x[idx], grid.y[idx], grid.z[idx] };
                const Vec3 normal = vec_normalize({ grid.nx[idx], grid.ny[idx], grid.nz[idx] });
                const Vec3 du = { grid.dxdu[idx], grid.dydu[idx], grid.dzdu[idx] };

                Vec3 t1, t2;
                build_tangent_frame(normal, du, t1, t2);

                // Launch from just off the surface along the outward normal
                // — reuses the same nudge RayTracingDriver sizes off this
                // patch's own curvature padding, so stage-1 intersect() can't
                // immediately re-detect the launch point itself.
                const Vec3 origin = { point.x + normal.x * eps,
                                       point.y + normal.y * eps,
                                       point.z + normal.z * eps };

                int best = 0;
                int n_errors = 0;
                for (int ti = 0; ti < theta_n; ++ti) {
                    // Solid-angle-uniform, cell-centered in cos(theta): mu is
                    // strictly inside (0,1), so theta never lands exactly on
                    // the pole or the grazing equator.
                    const double mu = (static_cast<double>(ti) + 0.5) / static_cast<double>(theta_n);
                    const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu * mu));

                    for (int pj = 0; pj < phi_n; ++pj) {
                        const double phi = (static_cast<double>(pj) + 0.5) * kTwoPi / static_cast<double>(phi_n);
                        const double cos_phi = std::cos(phi);
                        const double sin_phi = std::sin(phi);

                        const Vec3 dir = {
                            t1.x * sin_theta * cos_phi + t2.x * sin_theta * sin_phi + normal.x * mu,
                            t1.y * sin_theta * cos_phi + t2.y * sin_theta * sin_phi + normal.y * mu,
                            t1.z * sin_theta * cos_phi + t2.z * sin_theta * sin_phi + normal.z * mu
                        };

                        const RayTracingDriver::RayPath path = driver_.trace_ray(origin, dir, max_bounces);
                        if (path.status == RayTracingDriver::RayPath::Status::Error) {
                            ++n_errors;
                            continue;  // skip this direction, keep going
                        }
                        best = std::max(best, path.bounce_count);
                    }
                }
                max_bounces_at_point[idx] = best;
                error_count_at_point[idx] = n_errors;

                const long long done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
                if (done % print_every == 0 || done == nn) {
                    #pragma omp critical(bounce_map_progress)
                    {
                        print_progress(patch_num + 1, num_patches, done, nn);
                    }
                }
            }
            std::cout << '\n';  // finalize this patch's bar, move to the next line

            patch_grids_.push_back(std::move(grid));
            patch_max_bounces_.push_back(std::move(max_bounces_at_point));
            patch_error_counts_.push_back(std::move(error_count_at_point));
        }
    }

    // Writes the sampled range as one triangulated legacy ASCII VTK
    // PolyData surface: every sampled patch's position_grid_n x
    // position_grid_n grid triangulated the same way SurfacePatch's own
    // coarse grid is (two triangles per cell), all of them concatenated into
    // one point/polygon list, with per-point "MaxBounces" scalars. Call
    // run() first.
    void write_vtk(const std::string& filename) const {
        if (patch_grids_.empty()) {
            throw std::runtime_error("BounceMapDriver::write_vtk: no data — call run() first");
        }

        std::vector<Vec3> all_points;
        std::vector<int> all_bounces;
        std::vector<int> all_errors;
        struct Tri { int a, b, c; };
        std::vector<Tri> tris;

        for (size_t p = 0; p < patch_grids_.size(); ++p) {
            const auto& grid = patch_grids_[p];
            const auto& bounces = patch_max_bounces_[p];
            const auto& errors = patch_error_counts_[p];
            const int n = grid.n;
            const int base = static_cast<int>(all_points.size());

            for (int idx = 0; idx < n * n; ++idx) {
                all_points.push_back({ grid.x[idx], grid.y[idx], grid.z[idx] });
                all_bounces.push_back(bounces[idx]);
                all_errors.push_back(errors[idx]);
            }

            // Same diagonal split as SurfacePatch::intersect()'s coarse grid:
            // (p00,p10,p11) and (p00,p11,p01) per cell.
            for (int i = 0; i + 1 < n; ++i) {
                for (int j = 0; j + 1 < n; ++j) {
                    const int p00 = base + (i * n + j);
                    const int p10 = base + ((i + 1) * n + j);
                    const int p01 = base + (i * n + (j + 1));
                    const int p11 = base + ((i + 1) * n + (j + 1));
                    tris.push_back({ p00, p10, p11 });
                    tris.push_back({ p00, p11, p01 });
                }
            }
        }

        std::ofstream out(filename);
        if (!out.is_open()) {
            throw std::runtime_error("BounceMapDriver::write_vtk: cannot open output file: " + filename);
        }

        out << "# vtk DataFile Version 3.0\n";
        out << "Billiard-ball bounce map\n";
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
        out << "SCALARS MaxBounces int 1\nLOOKUP_TABLE default\n";
        for (int b : all_bounces) out << b << '\n';
        // Diagnostic: how many of this point's hemisphere directions errored
        // out (skipped, uncounted toward MaxBounces) rather than legitimately
        // escaping or getting trapped — see run()'s comment. A high count
        // here alongside MaxBounces == 0 means "most rays failed to trace",
        // not "this point genuinely has no bounces".
        out << "SCALARS ErrorCount int 1\nLOOKUP_TABLE default\n";
        for (int e : all_errors) out << e << '\n';
    }

    const RayTracingDriver& driver() const { return driver_; }

private:
    RayTracingDriver driver_;                                  // full scene (every *.txt in the directory)
    std::vector<std::unique_ptr<SurfacePatch>> sample_patches_; // just the range being sampled
    std::vector<SurfacePatch::SampledGrid> patch_grids_;
    std::vector<std::vector<int>> patch_max_bounces_;
    std::vector<std::vector<int>> patch_error_counts_;  // parallel to patch_max_bounces_; see run()

    static constexpr double kTwoPi = 6.283185307179586476925286766559;
    static constexpr int kProgressBarWidth = 30;

    // Relative tolerance for build_tangent_frame()'s degenerate-du fallback:
    // below this fraction of |du| itself, du is treated as (numerically)
    // parallel to the normal rather than trusted as a tangent direction.
    static constexpr double kTangentDegenerateRelTol = 1e-6;

    // Redraws a single-line progress bar in place (via '\r', no newline) for
    // the grid-point sweep of patch `patch_num_1based` out of `num_patches`.
    // Called from inside a #pragma omp critical section — must not be
    // called concurrently.
    static void print_progress(size_t patch_num_1based, size_t num_patches,
                                long long done, long long total) {
        const double frac = total > 0 ? static_cast<double>(done) / static_cast<double>(total) : 1.0;
        const int filled = static_cast<int>(frac * kProgressBarWidth);

        std::cout << '\r' << "Patch " << patch_num_1based << '/' << num_patches << " [";
        for (int i = 0; i < kProgressBarWidth; ++i) {
            std::cout << (i < filled ? '#' : '-');
        }
        std::cout << "] " << static_cast<int>(frac * 100.0) << "% ("
                   << done << '/' << total << " points)" << std::flush;
    }

    // Loads "<patch_directory><file_prefix><i>.txt" for i in
    // [first_index_1based, last_index_1based] into sample_patches_ — these
    // are used only as sampling sources (position + normal grids), never
    // added to driver_'s BVH (they're already part of the scene it loaded
    // from the directory scan). Throws on an empty/invalid range or a
    // missing/malformed file, same fail-loud behavior as RayTracingDriver's
    // own range constructor.
    void load_sample_patches(const std::string& patch_directory, const std::string& file_prefix,
                              int first_index_1based, int last_index_1based) {
        if (last_index_1based < first_index_1based) {
            throw std::invalid_argument(
                "BounceMapDriver: empty patch index range [" + std::to_string(first_index_1based) +
                ", " + std::to_string(last_index_1based) + "]");
        }

        sample_patches_.reserve(static_cast<size_t>(last_index_1based - first_index_1based + 1));
        for (int i = first_index_1based; i <= last_index_1based; ++i) {
            sample_patches_.push_back(
                std::make_unique<SurfacePatch>(patch_directory + file_prefix + std::to_string(i) + ".txt"));
        }
    }

    // Builds an orthonormal tangent frame (t1, t2) perpendicular to unit
    // normal n, with t1 the patch's own dX/du direction (`du`) projected
    // orthogonal to n and normalized — NOT picked off the global x/y/z axes.
    //
    // This matters because the hemisphere t1/t2 anchors is only swept
    // CONTINUOUSLY over the full phi range in the limit; run() actually
    // samples a finite theta_n x phi_n grid relative to this frame. A
    // reference axis chosen by comparing n against the global axes (the
    // previous approach) flips depending on which way the patch happens to
    // be rotated in the scene, even though the patch's own geometry hasn't
    // changed — silently handing two placements of the same patch two
    // different discrete direction sets. Billiard-ball bouncing is chaotic
    // enough that a different discrete direction set can produce noticeably
    // different bounce/error statistics even for an identical patch, purely
    // because of how it's oriented in world space. Building t1 from the
    // patch's own dX/du instead makes the sampled directions a property of
    // the surface, not of its placement, so run()'s results for a given
    // patch no longer depend on translation/rotation of the scene.
    //
    // Falls back to the old global-axis trick only in the degenerate case
    // where `du` is (numerically) parallel to n or zero-length — e.g. at a
    // parametrization singularity. At that one point, which axis gets
    // picked is genuinely arbitrary (there's no well-defined tangent
    // direction to be intrinsic to), so reproducibility isn't on the table
    // there anyway.
    static void build_tangent_frame(const Vec3& n, const Vec3& du, Vec3& t1, Vec3& t2) {
        const double du_len2 = vec_dot(du, du);
        const double du_dot_n = vec_dot(du, n);
        const Vec3 du_perp = { du.x - du_dot_n * n.x, du.y - du_dot_n * n.y, du.z - du_dot_n * n.z };
        const double du_perp_len2 = vec_dot(du_perp, du_perp);

        // Relative test (du_perp vs. du itself, not an absolute length)
        // so it works regardless of this patch's own dX/du scale. Written
        // as a product comparison rather than a ratio so du_len2 == 0
        // (du_perp_len2 is then 0 too) falls straight into the fallback
        // instead of a 0/0 division.
        if (du_perp_len2 > kTangentDegenerateRelTol * kTangentDegenerateRelTol * du_len2) {
            t1 = vec_normalize(du_perp);
        } else {
            const Vec3 a = (std::fabs(n.x) <= std::fabs(n.y) && std::fabs(n.x) <= std::fabs(n.z))
                               ? Vec3{ 1.0, 0.0, 0.0 }
                               : (std::fabs(n.y) <= std::fabs(n.z) ? Vec3{ 0.0, 1.0, 0.0 } : Vec3{ 0.0, 0.0, 1.0 });
            t1 = vec_normalize(vec_cross(n, a));
        }
        t2 = vec_cross(n, t1);  // unit already: n, t1 are orthonormal unit vectors
    }
};
