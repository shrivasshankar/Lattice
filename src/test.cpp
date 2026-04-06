#include "vector.h"
#include <iostream>

int main() {
    std::cout << "Testing vector library..." << std::endl;
    return 0;
}

void load_vectors(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open file: " + filepath);
    }

    file.read(reinterpret_cast<char*>(&num_vectors), sizeof(num_vectors));
    file.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    total_floats = static_cast<size_t>(num_vectors) * dimension;
    data.resize(total_floats);
    file.read(reinterpret_cast<char*>(data.data()), total_floats * sizeof(float));

    if (!file) {
        throw std::runtime_error("failed to read vector data from: " + filepath);
    }

    return data;
}
