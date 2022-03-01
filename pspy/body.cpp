#include "body.h"

#include <vector>

std::vector<std::shared_ptr<Body>> read_file(std::string path) {
    return read_xt(path);
}