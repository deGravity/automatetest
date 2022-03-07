#include "body.h"

#include <vector>

static std::string get_extension(std::string path) {
	size_t idx = path.find_last_of('.');
	std::string ext(path, idx + 1, path.size() - idx - 1);

	return ext;
}

std::vector<std::shared_ptr<Body>> read_file(std::string path) {
    std::string ext = get_extension(path);

    if (ext == "x_t" || ext == "xt") {
        return read_xt(path);
    }
    else if (ext == "step" || ext == "stp") {
        return read_step(path);
    }
    else {
        return std::vector<std::shared_ptr<Body>>();
    }
}
