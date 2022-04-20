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

    auto part = Part("C:/Users/Ben/research/crash.stp", options);
    //auto part = Part(TEST_STEP, options);
    std::cout << part._is_valid << std::endl;
    

    // Simple Test of KD-Tree

    //auto ipart_xt = ImplicitPart(TEST_X_T, 500, 5000, true);
    //auto ipart_step = ImplicitPart(TEST_STEP, 500, 5000, true);

    return 0;
}
