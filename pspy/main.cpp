#include <iostream>
#include <parasolid.h>

#include <vector>
#include <map>
#include <part.h>

using namespace pspy;

int main(int argc, char** argv) {
    PartOptions options;
    //options.onshape_style = false;
    //options.default_mcfs_only_face_axes = false;
    //options.num_uv_samples = 100;
    //options.collect_inferences = false;
    //options.default_mcfs = false;

    std::cout << "loading" << TEST_PART << std::endl;
    auto part = Part(TEST_PART, options);
    std::cout << "valid: " << part._is_valid << std::endl;
    return 0;
}
