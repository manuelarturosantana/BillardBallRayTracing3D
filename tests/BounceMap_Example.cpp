#include "../include/BounceMapDriver.hpp"

#include <iostream>
#include <string>

// Example usage of BounceMapDriver, mirroring src/main.cpp's use of
// RayTracingDriver. Not a pass/fail test (there's no ground truth to check
// against) — run it against real IFGF-RP patch geometry and inspect the
// output .vtk file (e.g. in ParaView, colored by "MaxBounces") to see how
// deep the geometry traps rays launched from a given range of patches.
//
//   ./bounce_map_example <patch_directory> <file_prefix> <first_index> <last_index> <output_vtk>
//
// e.g.:
//   ./bounce_map_example /scratch/msantana/PatchFolders/Nacelle/ Nacelle- 1 4 /scratch/msantana/RayTracing/bounce_map.vtk
int main(int argc, char** argv) {
    // --- Geometry ------------------------------------------------------
    // Directory of IFGF-RP patch ".txt" files — EVERY patch in here becomes
    // part of the traced scene (BounceMapDriver loads the whole directory
    // for the BVH), but only the [first_index, last_index] range below is
    // actually sampled (grid points + launch directions).
    // std::string patch_directory =
    //     (argc > 1) ? argv[1] : "/scratch/msantana/PatchFolders/Nacelle/";
    // std::string patch_directory =
    //     (argc > 1) ? argv[1] : "/scratch/msantana/PatchFoldersRefined/PlaneWithNacelleRefinedReordered/";
    std::string patch_directory =
        (argc > 1) ? argv[1] : "/scratch/msantana/PatchFoldersRefined/deleteme/";
    std::string file_prefix = (argc > 2) ? argv[2] : "patch_";
    int first_index = (argc > 3) ? std::stoi(argv[3]) : 1;
    int last_index  = (argc > 4) ? std::stoi(argv[4]) : 134;

    // Output VTK PolyData surface — open it in ParaView and color by
    // "MaxBounces" to see the bounce map.
    std::string output_vtk =
        (argc > 5) ? argv[5] : "/scratch/msantana/RayTracing/bounce_map2.vtk";

    // --- Sampling resolution ---------------------------------------------
    const int position_grid_n = 6;  // n x n position samples per patch
    const int theta_n = 6;           // polar-angle (solid-angle-uniform) samples
    const int phi_n = 9;            // azimuth samples
    const int max_bounces = 100;     // per-direction bounce cap (Trapped beyond this)

    try {
        BounceMapDriver driver(patch_directory, file_prefix, first_index, last_index);
        driver.run(position_grid_n, theta_n, phi_n, max_bounces);
        driver.write_vtk(output_vtk);

        std::cout << "Sampled patches " << first_index << "-" << last_index
                  << " (" << position_grid_n << "x" << position_grid_n << " grid, "
                  << theta_n << "x" << phi_n << " directions/point)\n";
        std::cout << "Wrote bounce map to " << output_vtk << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
