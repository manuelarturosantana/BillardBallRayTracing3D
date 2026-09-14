#include "RayTracingDriver.hpp"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    // --- Geometry ------------------------------------------------------
    // Directory of IFGF-RP patch ".txt" files describing the scattering
    // surface. Override via argv[1], e.g.:
    //   ./trace_scene ../../ThreeDResonances/Grids/TestPatch
    std::string patch_directory =
        (argc > 1) ? argv[1] : "/scratch/msantana/PatchFolders/Nacelle";

    // Output VTK PolyData file — open it in ParaView to see the traced paths.
    std::string output_vtk = (argc > 2) ? argv[2] : "/scratch/msantana/RayTracing/ray_traces.vtk";

    // --- Rays --------------------------------------------------------------
    // Edit these to set up whatever incident rays you want to trace.
    // Paired 1:1: ray i starts at sources[i] and travels along directions[i].
    std::vector<Vec3> sources = {
        { 1,  0.5,  3.2 },
        { 0.5, 0.0,-1},
        { 0.6, 0.0,-1},
    };
    std::vector<Vec3> directions = {
        { 1,  -1,   -1  },
        {-1, 0, 0},
        {0, 1, 1},
    };

    try {
        RayTracingDriver driver(patch_directory);
        driver.set_rays(sources, directions);
        driver.run(/*max_bounces=*/300);

        int escaped = 0, trapped = 0, errored = 0;
        const auto& paths = driver.paths();
        for (size_t i = 0; i < paths.size(); ++i) {
            const auto& path = paths[i];
            std::cout << "Ray " << i << ": " << path.bounce_count << " bounce(s), status = ";
            switch (path.status) {
                case RayTracingDriver::RayPath::Status::Escaped:
                    std::cout << "escaped";
                    ++escaped;
                    break;
                case RayTracingDriver::RayPath::Status::Trapped:
                    std::cout << "trapped (hit bounce cap)";
                    ++trapped;
                    break;
                case RayTracingDriver::RayPath::Status::Error:
                    std::cout << "error (" << path.error_message << ")";
                    ++errored;
                    break;
            }
            std::cout << "\n";
        }
        std::cout << escaped << " escaped, " << trapped << " trapped, " << errored
                  << " errored, out of " << paths.size() << " rays.\n";

        driver.write_vtk(output_vtk);
        std::cout << "Wrote ray traces to " << output_vtk << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
