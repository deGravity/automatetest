#include <iostream>
#include <parasolid.h>

#include <vector>
#include <map>
#include <part.h>

#include "neoflann.h"


int main(int argc, char** argv) {
    /*
    PartOptions options;
    options.onshape_style = false;
    options.default_mcfs_only_face_axes = false;
    options.num_uv_samples = 0;
    options.num_random_samples = 0;
    options.num_sdf_samples = 5000;
    options.sdf_sample_quality = 5000;
    options.collect_inferences = false;
    options.default_mcfs = false;

    auto part = Part(TEST_PART, options);
    */

    // Simple Test of KD-Tree

    Eigen::MatrixXd points(4, 2);
    points <<
        0, 0,
        1, 0,
        1, 1,
        0, 1;

    nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kdtree(2, points, 10);
    kdtree.index->buildIndex();
    double query_point[2];
    query_point[0] = 0.1;
    query_point[1] = 0.0;
    Eigen::Index out_index;
    double distance;
    kdtree.query(query_point, 1, &out_index, &distance);

    std::cout << distance;

    return 0;
}
