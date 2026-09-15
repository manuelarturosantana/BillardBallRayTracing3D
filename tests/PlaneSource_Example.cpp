#include "../include/PlaneSourceDriver.hpp"

#include <iostream>
#include <string>

// Example usage of PlaneSourceDriver, mirroring tests/BounceMap_Example.cpp's
// use of BounceMapDriver. Not a pass/fail test (there's no ground truth to
// check against) — run it against real IFGF-RP patch geometry and inspect
// the output .vtk file (e.g. in ParaView, colored by "HitCount") to see
// which patches a plane-wave-like source illuminates.
//
//   ./plane_source_example <patch_directory> <output_vtk>
//
// The plane's two corners, its fixed launch direction, and the nx x ny
// source grid are hardcoded below — edit them to match the geometry you're
// pointing at (the two corners must agree in exactly one coordinate; see
// PlaneSourceDriver::run()'s doc comment for how the remaining two axes map
// to nx/ny).
int main(int argc, char** argv) {
    // --- Geometry ------------------------------------------------------
    // Directory of IFGF-RP patch ".txt" files — every patch in here is part
    // of the traced scene, and every patch is written out (hit or not).
    std::string patch_directory =
        (argc > 1) ? argv[1] : "/scratch/msantana/PatchFolders/Nacelle/";

    // Output VTK PolyData surface — open it in ParaView and color by
    // "HitCount" to see which patches got illuminated.
    std::string output_vtk =
        (argc > 2) ? argv[2] : "/scratch/msantana/RayTracing/plane_source_hits.vtk";

    // --- Source plane -----------------------------------------------------
    // Axis-aligned: corner0/corner1 must match in exactly one coordinate —
    // here both have x = -5, so the plane spans y and z at x = -5, with nx
    // samples along y and ny samples along z (see PlaneSourceDriver::run()).
    //Nacelle
    // Vec3 corner0 = { -10.0, -10.0, 6.0};
    // Vec3 corner1 = { 10.0,  10.0,  6.0};
    // Vec3 direction = {0.0, 1.0, -1.0};  // fixed launch direction for every source point

    // const int nx = 300;
    // const int ny = 300;
    // Plane
    //   kx = cos(THE) sin(PHI), ky = sin(THE) sin(PHI), kz = cos(PHI).
// Note the plane faces the x- so waves hitting the front should point in the + x direction
// static const double PLANE_WAVE_THE = 0;          // azimuth, in [0, 2*pi)   // Up 30 degrees from x plus in z direction. 
// static const double PLANE_WAVE_PHI = M_PI / 3.0;  // polar,   in [0, pi] 
static const double THE = 0;          // azimuth, in [0, 2*pi)   // Up 30 degrees from x plus in -z direction. 
static const double PHI = 2.0 * M_PI / 3.0;  // polar,   in [0, pi]   

    Vec3 corner0 = { -5.0, -30.0, 8};
    Vec3 corner1 = { -5.0,  30.0, 112.5};
    Vec3 direction = {std::cos(THE) * std::sin(PHI), std::sin(THE) * std::sin(PHI), std::cos(PHI)};  // fixed launch direction for every source point

    std::cout << "direction " << direction.x << " " << direction.y << " " << direction.z << std::endl;
    // nx and ny are ordered according to the axis order.
    const int nx = 1040;
    const int ny = 1040;

    const int max_bounces = 300;  // per-ray bounce cap (Trapped beyond this)

    try {
        PlaneSourceDriver driver(patch_directory);
        driver.run(corner0, corner1, direction, nx, ny, max_bounces);
        driver.write_vtk(output_vtk);

        std::cout << "Traced " << (nx * ny) << " source points (" << nx << "x" << ny
                  << ") across " << driver.driver().patches().size() << " patches\n";
        std::cout << "Wrote patch hit counts to " << output_vtk << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
