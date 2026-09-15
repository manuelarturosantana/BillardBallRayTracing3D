#pragma once

#include "../ScatObject.hpp"
#include "../Utils/MollerTrumbore.hpp"
#include "../Utils/PatchInterpolation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// --- SurfacePatch -----------------------------------------------------------
// A single curved surface patch produced by the IFGF-RP geometry pipeline.
// On disk each patch is one ".txt" file: a tensor-product grid of
// interpolation nodes (u x v), the Cartesian coordinates of the surface at
// every node, plus first derivatives, surface normals and the area element.
//
// Construction loads one such file, then:
//   - computes the centroid and an axis-aligned bounding box (BVH needs
//     these up front);
//   - builds a coarse, equally-spaced grid over the reference square
//     [-1,1] x [-1,1] by Lagrange-interpolating the node data;
//   - measures, per coarse cell, how far the true surface bulges away from
//     that cell's flat approximation (compute_cell_padding()), and inflates
//     the bounding box by the worst case so BVH culling can't reject a
//     patch whose curvature the coarse grid understates.
//
// intersect() is two stages. Stage 1: each coarse grid square is split into
// two triangles, tested with Moller-Trumbore both as-is and pushed out by
// that cell's measured curvature margin along its normal (a ray can cross
// the true curved surface in the gap between a flat coarse chord and the
// real bulge — the margin closes that gap), closest hit wins. This only
// locates which coarse triangle the ray crosses. Stage 2: Newton's method
// refines that hit onto the true interpolated surface, solving
// X(u,v) - ray(t) = 0 for (u,v,t) starting from the winning triangle's
// (u,v) centroid and its stage-1 t. On failure to converge (iteration cap,
// singular Jacobian, or a solution outside the patch's [-1,1]x[-1,1]
// domain) it throws rather than silently falling back — see newton_refine().
//
// This header does NOT depend on the IFGF-RP headers directly, but it does
// pull in Utils/PatchInterpolation.hpp for the Lagrange interpolation, which
// (mirroring IFGF-RP's global.h) needs MKL for cblas_dgemm — any target that
// includes this header needs the same MKL/OpenMP/MPI link setup IFGF-RP uses
// (see IFGF-RP/CMakeLists.txt's GLOBAL_INTERFACE). The file format read here
// is mirrored from IFGF-RP's read_patch_file() in
// geometry_processing/PatchRefinement.cpp; keep the two in sync.

class SurfacePatch : public ScatObject {
public:
    // Raw patch data, exactly as it comes off disk. Flat arrays are stored
    // in the file's order: index = i + imax*(j + jmax*k), with k the fast
    // ... actually the file lists x, then y, then z, each as imax*jmax*kmax
    // values in the order they were written by the generator. We keep them
    // flat and untouched; helpers below index them.
    struct Data {
        int imax = 0;          // number of u nodes
        int jmax = 0;          // number of v nodes
        int kmax = 0;          // number of layers (1 for a surface)
        long long zoneID = 0;

        std::vector<double> uNodes;   // size imax
        std::vector<double> vNodes;   // size jmax

        // Barycentric weights for uNodes/vNodes, used by every Lagrange
        // interpolation done on this patch (e.g. building the coarse grid
        // below). Computed once, right after the nodes are read.
        std::vector<double> uWeights; // size imax
        std::vector<double> vWeights; // size jmax

        std::vector<double> x, y, z;  // size imax*jmax*kmax  (surface points)
        std::vector<int>    mask;     // size imax*jmax*kmax

        std::vector<double> dxdu, dydu, dzdu;   // du derivatives
        std::vector<double> dxdv, dydv, dzdv;   // dv derivatives
        std::vector<double> dS;                 // area element
        std::vector<double> nuX, nuY, nuZ;      // unit surface normal

        // Sharp-edge markers for the four patch sides [u=-1, u=+1, v=-1, v=+1].
        // Zero-filled when the file has no edge block.
        std::array<int, 4> edge_flags{ {0, 0, 0, 0} };

        long long point_count() const {
            return static_cast<long long>(imax) * jmax * kmax;
        }
    };

    // A coarse, equally-spaced sampling of the patch surface over the
    // reference square [-1,1] x [-1,1], built by Lagrange-interpolating the
    // node data (see compute_coarse_grid()). This is the mesh stage-1 of
    // intersect() tests rays against.
    struct CoarseGrid {
        int n = 0;                    // grid is n x n
        std::vector<double> u, v;     // size n each, equally spaced in [-1,1]
        std::vector<double> x, y, z;  // size n*n, row-major: idx = i*n + j

        // Per-cell curvature padding: how far the true interpolated surface
        // strays from this cell's flat bilinear estimate (the "sagitta"),
        // and a representative surface normal there. See
        // compute_cell_padding() — this is what keeps intersect()'s stage-1
        // test from missing a real crossing just because the coarse mesh is
        // flatter than the true curved patch.
        std::vector<double> cell_sagitta;  // size (n-1)*(n-1)
        std::vector<Vec3> cell_normal;     // size (n-1)*(n-1)

        Vec3 point(int i, int j) const {
            int idx = i * n + j;
            return { x[idx], y[idx], z[idx] };
        }

        int cell_index(int i, int j) const { return i * (n - 1) + j; }
    };

    static constexpr int kCoarseGridSize = 40;

    // Newton refinement (stage 2 of intersect()): iteration cap and the
    // convergence tolerance on the (u,v,t) update's step size, per iteration.
    static constexpr int kNewtonMaxIters = 20;
    static constexpr double kNewtonTol = 1e-3;

    // Load a patch from an explicit file path.
    explicit SurfacePatch(const std::string& path)
        : data_(load_patch_file(path))
    {
        compute_centroid_and_box();
        compute_coarse_grid(kCoarseGridSize);
        compute_cell_padding();
    }

    // Convenience overload matching IFGF-RP's directory + prefix + index
    // naming: files on disk are 1-based, so index 0 maps to "<prefix>1.txt".
    SurfacePatch(const std::string& directory,
                 const std::string& file_prefix,
                 long long index)
        : SurfacePatch(directory + file_prefix + std::to_string(index + 1) + ".txt")
    {}

    const Data& data() const { return data_; }
    const CoarseGrid& coarse_grid() const { return coarse_; }

    // Worst-case sagitta over every coarse cell (see compute_cell_padding()):
    // how far intersect()'s padded stage-1 test can sit from the true
    // surface. A ray reflecting off this patch needs to clear a self-
    // intersection nudge of at least this size, or the very next intersect()
    // call can re-detect the point it just left through the padded shell —
    // see RayTracingDriver's use of this.
    // WARNING: Note this could still possibly break if each subpatch is not well resolved. 
    // In particular the sagitta is made using the midpoint, and if the max curvature is far
    // from the midpoint this won't work. With well resolved patches this should be fine though.
    double max_sagitta() const { return max_sagitta_; }

    // ScatObject's per-object self-intersection clearance — this patch's own
    // worst-case sagitta, not any scene-wide value. See ScatObject::
    // self_intersect_padding() and RayTracingDriver::self_intersect_epsilon().
    double self_intersect_padding() const override { return max_sagitta_; }

    // A general-purpose n x n sampling of the patch surface (position AND
    // unit normal) over the reference square [-1,1] x [-1,1], via Lagrange
    // interpolation — independent of, and typically a different resolution
    // than, the internal coarse_ grid intersect() uses for hit-testing.
    // Callers that want to sample the patch itself (e.g. to launch rays
    // from its surface, as BounceMapDriver does) should use this, not
    // coarse_grid().
    struct SampledGrid {
        int n = 0;
        std::vector<double> u, v;         // size n each, equally spaced in [-1,1]
        std::vector<double> x, y, z;      // size n*n, row-major: idx = i*n + j
        std::vector<double> nx, ny, nz;   // size n*n, unit normal at each point

        Vec3 point(int i, int j) const {
            int idx = i * n + j;
            return { x[idx], y[idx], z[idx] };
        }
        Vec3 normal(int i, int j) const {
            int idx = i * n + j;
            return { nx[idx], ny[idx], nz[idx] };
        }
    };

    SampledGrid sample_grid(int n) const {
        SampledGrid g;
        g.n = n;
        g.u.resize(n);
        g.v.resize(n);
        for (int i = 0; i < n; ++i) {
            const double t = (n == 1) ? 0.0 : -1.0 + 2.0 * static_cast<double>(i) / (n - 1);
            g.u[i] = t;
            g.v[i] = t;
        }

        const size_t nn = static_cast<size_t>(n) * n;
        g.x.resize(nn);  g.y.resize(nn);  g.z.resize(nn);
        g.nx.resize(nn); g.ny.resize(nn); g.nz.resize(nn);

        lagrange_interpolation_2D(data_.uNodes, data_.vNodes, data_.uWeights, data_.vWeights,
                                   data_.x, g.u, g.v, g.x.data());
        lagrange_interpolation_2D(data_.uNodes, data_.vNodes, data_.uWeights, data_.vWeights,
                                   data_.y, g.u, g.v, g.y.data());
        lagrange_interpolation_2D(data_.uNodes, data_.vNodes, data_.uWeights, data_.vWeights,
                                   data_.z, g.u, g.v, g.z.data());
        lagrange_interpolation_2D(data_.uNodes, data_.vNodes, data_.uWeights, data_.vWeights,
                                   data_.nuX, g.u, g.v, g.nx.data());
        lagrange_interpolation_2D(data_.uNodes, data_.vNodes, data_.uWeights, data_.vWeights,
                                   data_.nuY, g.u, g.v, g.ny.data());
        lagrange_interpolation_2D(data_.uNodes, data_.vNodes, data_.uWeights, data_.vWeights,
                                   data_.nuZ, g.u, g.v, g.nz.data());

        // Lagrange-interpolating a unit vector field doesn't generally
        // preserve unit length; renormalize per point.
        for (size_t k = 0; k < nn; ++k) {
            const Vec3 nrm = vec_normalize({ g.nx[k], g.ny[k], g.nz[k] });
            g.nx[k] = nrm.x; g.ny[k] = nrm.y; g.nz[k] = nrm.z;
        }

        return g;
    }

    // Stage 1: brute-force test against the coarse grid. Each grid square
    // (i,j)-(i+1,j)-(i,j+1)-(i+1,j+1) is split into two triangles; since the
    // true patch is curved and the coarse mesh is flat, a real crossing can
    // fall in the gap between the flat chord and the true surface bulge
    // (compute_cell_padding() measures that gap per cell), so each triangle
    // is tested as-is AND pushed out ± that cell's sagitta along its local
    // normal — any of the three catching the ray is enough to seed stage 2.
    // Closest hit over all cells/paddings wins. Stage 2: Newton-refine that
    // hit onto the true interpolated surface (see newton_refine()).
    std::optional<Hit> intersect(const Ray& ray) const override {
        const int n = coarse_.n;
        std::optional<Hit> best;
        int best_i = -1, best_j = -1;
        bool best_second_triangle = false;

        auto keep_closer = [&](const std::optional<Hit>& candidate, int i, int j, bool second_triangle) {
            if (candidate && (!best || candidate->t < best->t)) {
                best = candidate;
                best_i = i;
                best_j = j;
                best_second_triangle = second_triangle;
            }
        };

        auto offset = [](const Vec3& p, const Vec3& normal, double d) -> Vec3 {
            return { p.x + normal.x * d, p.y + normal.y * d, p.z + normal.z * d };
        };

        for (int i = 0; i + 1 < n; ++i) {
            for (int j = 0; j + 1 < n; ++j) {
                Vec3 p00 = coarse_.point(i,     j);
                Vec3 p10 = coarse_.point(i + 1, j);
                Vec3 p01 = coarse_.point(i,     j + 1);
                Vec3 p11 = coarse_.point(i + 1, j + 1);

                const int idx = coarse_.cell_index(i, j);
                const double s = coarse_.cell_sagitta[idx];
                const Vec3& cn = coarse_.cell_normal[idx];

                // Diagonal p00-p11 splits the square into (p00,p10,p11) and
                // (p00,p11,p01). Test the flat chord (sign 0) plus both
                // curvature-padded copies — the true surface can bulge to
                // either side of the coarse chord, and we don't know which.
                for (double sign : { 0.0, 1.0, -1.0 }) {
                    const double d = sign * s;
                    Vec3 q00 = offset(p00, cn, d), q10 = offset(p10, cn, d);
                    Vec3 q01 = offset(p01, cn, d), q11 = offset(p11, cn, d);

                    keep_closer(moller_trumbore(ray, q00, q10, q11), i, j, false);
                    keep_closer(moller_trumbore(ray, q00, q11, q01), i, j, true);
                }
            }
        }

        if (!best) return std::nullopt;

        double u0, v0;
        triangle_uv_centroid(best_i, best_j, best_second_triangle, u0, v0);
        return newton_refine(ray, u0, v0, best->t);
    }

private:
    Data data_;
    CoarseGrid coarse_;
    double max_sagitta_ = 0.0;

    // Reads one patch file. Mirrors IFGF-RP read_patch_file(): a fixed
    // sequence of whitespace-separated numbers, with an optional trailing
    // 4-value sharp-edge block.
    static Data load_patch_file(const std::string& path) {
        std::ifstream fin(path);
        if (!fin.is_open()) {
            throw std::runtime_error("SurfacePatch: cannot open patch file: " + path);
        }

        Data d;
        fin >> d.imax >> d.jmax >> d.kmax >> d.zoneID;
        if (!fin) {
            throw std::runtime_error("SurfacePatch: bad header in patch file: " + path);
        }

        const long long n = d.point_count();

        auto read_vec = [&](std::vector<double>& v, long long count) {
            v.resize(count);
            for (long long i = 0; i < count; ++i) fin >> v[i];
        };

        read_vec(d.uNodes, d.imax);
        read_vec(d.vNodes, d.jmax);

        d.uWeights = barycentric_weights(d.uNodes);
        d.vWeights = barycentric_weights(d.vNodes);

        read_vec(d.x, n);
        read_vec(d.y, n);
        read_vec(d.z, n);

        d.mask.resize(n);
        for (long long i = 0; i < n; ++i) fin >> d.mask[i];

        read_vec(d.dxdu, n);
        read_vec(d.dydu, n);
        read_vec(d.dzdu, n);
        read_vec(d.dxdv, n);
        read_vec(d.dydv, n);
        read_vec(d.dzdv, n);
        read_vec(d.dS, n);
        read_vec(d.nuX, n);
        read_vec(d.nuY, n);
        read_vec(d.nuZ, n);

        if (!fin) {
            throw std::runtime_error("SurfacePatch: truncated patch file: " + path);
        }

        // Optional trailing sharp-edge block: present or clean EOF, nothing else.
        fin >> std::ws;
        if (!fin.eof()) {
            if (!(fin >> d.edge_flags[0] >> d.edge_flags[1]
                      >> d.edge_flags[2] >> d.edge_flags[3])) {
                throw std::runtime_error(
                    "SurfacePatch: malformed sharp-edge data in patch file: " + path);
            }
        }

        return d;
    }

    // Centroid = mean of the surface node positions. Bounding box = tight
    // min/max over those same node positions — but see compute_cell_padding(),
    // called after this from the constructor, which inflates box_ by the
    // patch's worst-case coarse-grid curvature margin. The dense node grid
    // itself (imax x jmax, typically much finer than the 15x15 coarse grid)
    // is assumed to already hug the true surface closely enough that its own
    // node-to-node bulge is negligible next to that coarse-grid margin.
    void compute_centroid_and_box() {
        const long long n = data_.point_count();
        if (n == 0) {
            throw std::runtime_error("SurfacePatch: patch has no points");
        }

        double sx = 0.0, sy = 0.0, sz = 0.0;
        AABB box;  // starts inverted (+inf / -inf)

        for (long long i = 0; i < n; ++i) {
            const double px = data_.x[i];
            const double py = data_.y[i];
            const double pz = data_.z[i];

            sx += px; sy += py; sz += pz;

            box.min.x = std::min(box.min.x, px);
            box.min.y = std::min(box.min.y, py);
            box.min.z = std::min(box.min.z, pz);
            box.max.x = std::max(box.max.x, px);
            box.max.y = std::max(box.max.y, py);
            box.max.z = std::max(box.max.z, pz);
        }

        const double inv_n = 1.0 / static_cast<double>(n);
        centroid_ = { sx * inv_n, sy * inv_n, sz * inv_n };
        box_ = box;
    }

    // Builds coarse_: an n x n equally-spaced sampling of the patch over the
    // reference square [-1,1] x [-1,1], via 2D Lagrange interpolation of the
    // node data using the barycentric weights computed at load time.
    //
    // Assumes kmax == 1 (a surface, not a volume layer stack) — x/y/z are
    // then exactly an imax x jmax grid, which is what lagrange_interpolation_2D
    // expects as its fNodes argument (row-major, u-index major).
    void compute_coarse_grid(int n) {
        coarse_.n = n;
        coarse_.u.resize(n);
        coarse_.v.resize(n);
        for (int i = 0; i < n; ++i) {
            double t = (n == 1) ? 0.0 : -1.0 + 2.0 * static_cast<double>(i) / (n - 1);
            coarse_.u[i] = t;
            coarse_.v[i] = t;
        }

        coarse_.x.resize(static_cast<size_t>(n) * n);
        coarse_.y.resize(static_cast<size_t>(n) * n);
        coarse_.z.resize(static_cast<size_t>(n) * n);

        lagrange_interpolation_2D(data_.uNodes, data_.vNodes, data_.uWeights, data_.vWeights,
                                   data_.x, coarse_.u, coarse_.v, coarse_.x.data());
        lagrange_interpolation_2D(data_.uNodes, data_.vNodes, data_.uWeights, data_.vWeights,
                                   data_.y, coarse_.u, coarse_.v, coarse_.y.data());
        lagrange_interpolation_2D(data_.uNodes, data_.vNodes, data_.uWeights, data_.vWeights,
                                   data_.z, coarse_.u, coarse_.v, coarse_.z.data());
    }

    // For each coarse cell, measures how far the true interpolated surface
    // at the cell's parameter-space midpoint strays from the flat bilinear
    // estimate of its four coarse corners (the "sagitta"), and records the
    // surface normal there. intersect() pads its stage-1 triangle test by
    // this amount along the normal so a ray crossing the true curved
    // surface isn't missed just because it slips past the (flatter) coarse
    // chord. The worst-case sagitta over the whole patch also inflates
    // box_, for the same reason at the BVH broad-phase level.
    void compute_cell_padding() {
        const int n = coarse_.n;
        const int cells = (n - 1) * (n - 1);
        coarse_.cell_sagitta.assign(cells, 0.0);
        coarse_.cell_normal.assign(cells, Vec3{ 0.0, 0.0, 0.0 });

        std::vector<double> lu, lv;
        double max_sagitta = 0.0;

        for (int i = 0; i + 1 < n; ++i) {
            for (int j = 0; j + 1 < n; ++j) {
                const double u_mid = 0.5 * (coarse_.u[i] + coarse_.u[i + 1]);
                const double v_mid = 0.5 * (coarse_.v[j] + coarse_.v[j + 1]);

                lagrange_basis_1d(data_.uNodes, data_.uWeights, u_mid, lu);
                lagrange_basis_1d(data_.vNodes, data_.vWeights, v_mid, lv);

                Vec3 true_mid = { eval_scalar_field(lu, lv, data_.x),
                                   eval_scalar_field(lu, lv, data_.y),
                                   eval_scalar_field(lu, lv, data_.z) };
                Vec3 normal_mid = vec_normalize({ eval_scalar_field(lu, lv, data_.nuX),
                                                   eval_scalar_field(lu, lv, data_.nuY),
                                                   eval_scalar_field(lu, lv, data_.nuZ) });

                const Vec3 p00 = coarse_.point(i,     j);
                const Vec3 p10 = coarse_.point(i + 1, j);
                const Vec3 p01 = coarse_.point(i,     j + 1);
                const Vec3 p11 = coarse_.point(i + 1, j + 1);
                const Vec3 bilinear_mid = { 0.25 * (p00.x + p10.x + p01.x + p11.x),
                                             0.25 * (p00.y + p10.y + p01.y + p11.y),
                                             0.25 * (p00.z + p10.z + p01.z + p11.z) };

                const Vec3 diff = vec_sub(true_mid, bilinear_mid);
                const double sagitta = std::sqrt(vec_dot(diff, diff));

                const int idx = coarse_.cell_index(i, j);
                coarse_.cell_sagitta[idx] = sagitta;
                coarse_.cell_normal[idx] = normal_mid;
                max_sagitta = std::max(max_sagitta, sagitta);
            }
        }

        box_.min.x -= max_sagitta; box_.min.y -= max_sagitta; box_.min.z -= max_sagitta;
        box_.max.x += max_sagitta; box_.max.y += max_sagitta; box_.max.z += max_sagitta;

        max_sagitta_ = max_sagitta;
    }

    // (u,v) centroid of one coarse-grid triangle, straight from the coarse
    // grid's own parameter axes — coarse_.u[i]/coarse_.v[j] are exactly the
    // (u,v) the corner coarse_.point(i,j) was interpolated at. Matches the
    // two triangles intersect() splits each square into: (p00,p10,p11) for
    // second_triangle == false, (p00,p11,p01) for true.
    void triangle_uv_centroid(int i, int j, bool second_triangle, double& u0, double& v0) const {
        const double u_i = coarse_.u[i], u_ip1 = coarse_.u[i + 1];
        const double v_j = coarse_.v[j], v_jp1 = coarse_.v[j + 1];

        if (!second_triangle) {
            u0 = (u_i + u_ip1 + u_ip1) / 3.0;
            v0 = (v_j + v_j + v_jp1) / 3.0;
        } else {
            u0 = (u_i + u_i + u_ip1) / 3.0;
            v0 = (v_j + v_jp1 + v_jp1) / 3.0;
        }
    }

    // Barycentric Lagrange interpolation weights for evaluating at a single
    // parameter value x, given one axis's nodes and its precomputed
    // barycentric_weights(). Exact one-hot weights when x lands on a node
    // (matching lagrange_interpolation_2D's own on-node handling) — this is
    // the same formula, just for one point instead of a batch, so repeated
    // evaluations during Newton iteration don't pay for MKL's dgemm path.
    static void lagrange_basis_1d(const std::vector<double>& nodes,
                                   const std::vector<double>& weights,
                                   double x, std::vector<double>& basis) {
        const int N = static_cast<int>(nodes.size());
        basis.assign(N, 0.0);

        for (int i = 0; i < N; ++i) {
            if (std::fabs(x - nodes[i]) < EQUAL_TOL) {
                basis[i] = 1.0;
                return;
            }
        }

        double denom = 0.0;
        for (int i = 0; i < N; ++i) {
            double li = weights[i] / (x - nodes[i]);
            basis[i] = li;
            denom += li;
        }
        for (int i = 0; i < N; ++i) basis[i] /= denom;
    }

    // Evaluates one node-grid field at the point whose basis weights are
    // (lu, lv). field is the same imax*jmax flat layout as data_.x etc.
    double eval_scalar_field(const std::vector<double>& lu, const std::vector<double>& lv,
                              const std::vector<double>& field) const {
        const int jmax = data_.jmax;
        double sum = 0.0;
        for (int i = 0; i < data_.imax; ++i) {
            const double li = lu[i];
            if (li == 0.0) continue;
            const double* row = &field[static_cast<size_t>(i) * jmax];
            double row_sum = 0.0;
            for (int j = 0; j < jmax; ++j) row_sum += lv[j] * row[j];
            sum += li * row_sum;
        }
        return sum;
    }

    // Solves the 3x3 linear system J*x = b via Cramer's rule. Returns false
    // (x left untouched) if J is numerically singular.
    static bool solve3x3(const double J[3][3], const Vec3& b, double x[3]) {
        const double a11 = J[0][0], a12 = J[0][1], a13 = J[0][2];
        const double a21 = J[1][0], a22 = J[1][1], a23 = J[1][2];
        const double a31 = J[2][0], a32 = J[2][1], a33 = J[2][2];

        const double det = a11 * (a22 * a33 - a23 * a32)
                          - a12 * (a21 * a33 - a23 * a31)
                          + a13 * (a21 * a32 - a22 * a31);
        if (std::fabs(det) < 1e-14) return false;

        const double bx = b.x, by = b.y, bz = b.z;

        const double det0 = bx * (a22 * a33 - a23 * a32) - a12 * (by * a33 - a23 * bz) + a13 * (by * a32 - a22 * bz);
        const double det1 = a11 * (by * a33 - a23 * bz) - bx * (a21 * a33 - a23 * a31) + a13 * (a21 * bz - by * a31);
        const double det2 = a11 * (a22 * bz - by * a32) - a12 * (a21 * bz - by * a31) + bx * (a21 * a32 - a22 * a31);

        x[0] = det0 / det;
        x[1] = det1 / det;
        x[2] = det2 / det;
        return true;
    }

    // Stage 2 of intersect(): Newton's method on (u,v,t), solving
    //   F(u,v,t) = X(u,v) - (ray.origin + t*ray.dir) = 0
    // Starting from the winning coarse triangle's (u,v) centroid and its
    // stage-1 t. Position and its u/v derivatives are interpolated from the
    // per-node x/y/z and dxdu.../dxdv... fields at the current (u,v); the
    // Jacobian's columns are dF/du = dX/du, dF/dv = dX/dv, dF/dt = -ray.dir.
    //
    // Per the "report an error" choice: this throws std::runtime_error
    // rather than falling back to the coarse hit if the Jacobian goes
    // singular, the iteration cap is hit before the step size drops below
    // kNewtonTol, or the converged (u,v) lands outside the patch's
    // [-1,1]x[-1,1] domain.
    std::optional<Hit> newton_refine(const Ray& ray, double u0, double v0, double t0) const {
        double u = u0, v = v0, t = t0;
        std::vector<double> lu, lv;
        double step_norm = std::numeric_limits<double>::infinity();

        for (int iter = 0; iter < kNewtonMaxIters; ++iter) {
            lagrange_basis_1d(data_.uNodes, data_.uWeights, u, lu);
            lagrange_basis_1d(data_.vNodes, data_.vWeights, v, lv);

            Vec3 X    = { eval_scalar_field(lu, lv, data_.x),    eval_scalar_field(lu, lv, data_.y),    eval_scalar_field(lu, lv, data_.z) };
            Vec3 dXdu = { eval_scalar_field(lu, lv, data_.dxdu), eval_scalar_field(lu, lv, data_.dydu), eval_scalar_field(lu, lv, data_.dzdu) };
            Vec3 dXdv = { eval_scalar_field(lu, lv, data_.dxdv), eval_scalar_field(lu, lv, data_.dydv), eval_scalar_field(lu, lv, data_.dzdv) };

            Vec3 R = { ray.origin.x + t * ray.dir.x,
                       ray.origin.y + t * ray.dir.y,
                       ray.origin.z + t * ray.dir.z };
            Vec3 F = vec_sub(X, R);

            const double J[3][3] = {
                { dXdu.x, dXdv.x, -ray.dir.x },
                { dXdu.y, dXdv.y, -ray.dir.y },
                { dXdu.z, dXdv.z, -ray.dir.z }
            };

            double delta[3];
            if (!solve3x3(J, F, delta)) {
                throw std::runtime_error(
                    "SurfacePatch::intersect: singular Jacobian during Newton refinement "
                    "(zoneID " + std::to_string(data_.zoneID) + "), likely a grazing ray");
            }

            u -= delta[0];
            v -= delta[1];
            t -= delta[2];

            step_norm = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
            if (step_norm < kNewtonTol) break;
        }

        if (step_norm >= kNewtonTol) {
            throw std::runtime_error(
                "SurfacePatch::intersect: Newton refinement failed to converge in " +
                std::to_string(kNewtonMaxIters) + " iterations (zoneID " +
                std::to_string(data_.zoneID) + "), final step norm " + std::to_string(step_norm));
        }

        constexpr double kDomainTol = 1e-6;
        if (u < -1.0 - kDomainTol || u > 1.0 + kDomainTol ||
            v < -1.0 - kDomainTol || v > 1.0 + kDomainTol) {
            throw std::runtime_error(
                "SurfacePatch::intersect: Newton refinement converged outside the patch "
                "domain (u=" + std::to_string(u) + ", v=" + std::to_string(v) +
                ", zoneID " + std::to_string(data_.zoneID) + ")");
        }

        // Final evaluation at the converged (u,v): exact surface point plus
        // the interpolated (then renormalized) surface normal.
        lagrange_basis_1d(data_.uNodes, data_.uWeights, u, lu);
        lagrange_basis_1d(data_.vNodes, data_.vWeights, v, lv);

        Hit hit;
        hit.t = t;
        hit.point = { eval_scalar_field(lu, lv, data_.x), eval_scalar_field(lu, lv, data_.y), eval_scalar_field(lu, lv, data_.z) };
        hit.normal = vec_normalize({ eval_scalar_field(lu, lv, data_.nuX),
                                      eval_scalar_field(lu, lv, data_.nuY),
                                      eval_scalar_field(lu, lv, data_.nuZ) });
        return hit;
    }
};
