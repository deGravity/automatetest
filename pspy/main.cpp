#include <iostream>
#include <parasolid.h>

#include <vector>
#include <map>
#include "part.h"

#include "implicit_part.h"

using namespace pspy;

int main(int argc, char** argv) {
    
    PartOptions options;
    options.normalize = false;
    options.transform = false;
    options.onshape_style = false;
    options.default_mcfs_only_face_axes = false;
    options.num_uv_samples = 0;
    options.num_random_samples = 0;
    options.num_sdf_samples = 0;//5000;
    options.sdf_sample_quality = 0;//5000;
    options.collect_inferences = false;
    options.default_mcfs = false;
    options.tesselate = false;

    auto part_x_t = Part(TEST_X_T, options);
    auto part_step = Part(TEST_STEP, options);
    std::cout << part_x_t._is_valid << std::endl;
    std::cout << part_step._is_valid << std::endl;
    

    return 0;
}
