#include "../include/BounceMapDriver.hpp"

#include <iostream>
#include <string>

// Example usage of BounceMapDriver, mirroring src/main.cpp's use of
// RayTracingDriver. Not a pass/fail test (there's no ground truth to check
// against) — run it against real IFGF-RP patch geometry and inspect the
// output .vtk file (e.g. in ParaView, colored by "MaxBounces") to see how
// deep the geometry traps rays launched from a given range of patches.
//
//   ./bounce_map_example <patch_directory> <file_prefix> <first_index> <last_index> <output_vtk> [best_rays_dir]
//
// e.g.:
//   ./bounce_map_example /scratch/msantana/PatchFolders/Nacelle/ Nacelle- 1 4 /scratch/msantana/RayTracing/bounce_map.vtk
//
// Passing a 6th argument additionally writes, per sampled patch, the single
// best (most-bounces) ray launched from each of that patch's grid points as
// its own .vtk line file into that directory (created if needed) — see
// BounceMapDriver::write_best_ray_vtks().
int main(int argc, char** argv) {
    // --- Geometry ------------------------------------------------------
    // Directory of IFGF-RP patch ".txt" files — EVERY patch in here becomes
    // part of the traced scene (BounceMapDriver loads the whole directory
    // for the BVH), but only the [first_index, last_index] range below is
    // actually sampled (grid points + launch directions).
    // std::string patch_directory =
    //     (argc > 1) ? argv[1] : "/scratch/msantana/PatchFolders/Nacelle/";
    std::string patch_directory =
        (argc > 1) ? argv[1] : "/scratch/msantana/PatchFoldersRefined/PlaneWithNacelleFirst/";
    // std::string patch_directory =
    //     (argc > 1) ? argv[1] : "/scratch/msantana/PatchFoldersRefined/deleteme/";
    std::string file_prefix = (argc > 2) ? argv[2] : "patch_";
    int first_index = (argc > 3) ? std::stoi(argv[3]) : 1;
    int last_index  = (argc > 4) ? std::stoi(argv[4]) : 134;

    // Output VTK PolyData surface — open it in ParaView and color by
    // "MaxBounces" to see the bounce map.
    std::string output_vtk =
        (argc > 5) ? argv[5] : "/scratch/msantana/RayTracing/bounce_map_60_more_triangles.vtk";

    // Optional: directory to write per-patch "best ray per grid point" VTK
    // files into. Left empty (the default) skips capturing/writing them.
    std::string best_rays_dir = (argc > 6) ? argv[6] : "/scratch/msantana/RayTracing/RaysMoreTriangles/";

    // --- Sampling resolution ---------------------------------------------
    const int position_grid_n = 10;  // n x n position samples per patch
    const int theta_n = 10;           // polar-angle (solid-angle-uniform) samples
    const int phi_n = 10;            // azimuth samples
    const double cone_angle_deg = 180.0;  // full apex angle about the normal; 180 = whole hemisphere
    const int max_bounces = 100;     // per-direction bounce cap (Trapped beyond this)

    const bool capture_best_rays = !best_rays_dir.empty();

    try {
        BounceMapDriver driver(patch_directory, file_prefix, first_index, last_index);
        driver.run(position_grid_n, theta_n, phi_n, cone_angle_deg, max_bounces, capture_best_rays);
        driver.write_vtk(output_vtk);

        std::cout << "Sampled patches " << first_index << "-" << last_index
                  << " (" << position_grid_n << "x" << position_grid_n << " grid, "
                  << theta_n << "x" << phi_n << " directions/point)\n";
        std::cout << "Wrote bounce map to " << output_vtk << "\n";

        if (capture_best_rays) {
            driver.write_best_ray_vtks(best_rays_dir);
            std::cout << "Wrote per-patch best-ray files to " << best_rays_dir << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
