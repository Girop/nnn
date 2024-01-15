#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

inline void displayProgressBar(unsigned int progress, unsigned int total, unsigned int width = 50) {
    float percentage = static_cast<float>(progress) / total;
    unsigned int barWidth = percentage * width;

    std::cout << "[";
    for (unsigned int i = 0; i < width; i++) {
        std::cout << (i < barWidth ? "=" : " ");
    }

    std::cout << "] " << std::fixed << std::setprecision(1) << percentage * 100.0 << "%\r";
    std::cout.flush();
}


inline void save_to_ppm(std::string const& filename, std::vector<std::vector<float>> const& image) {
    auto ppm_file = std::ofstream(filename, std::ios::out | std::ios::binary);

    if (!ppm_file.is_open()) {
        std::cerr << "Unable to open file for writing" << std::endl;
        exit(1);
    }

    const int width = static_cast<int>(image[0].size());
    const int height = static_cast<int>(image.size());

    ppm_file << "P1\n" << width << " " << height << "\n";

    for (const auto& row : image) {
        for (bool pixel : row) {
            ppm_file << pixel << ' ';
        }
       ppm_file.put('\n');
    }
}

template<std::size_t RowLength, typename T>
inline std::vector<std::vector<T>> unflatten_image(std::vector<T> image) {
    std::vector<std::vector<T>> result;
    std::vector<T> last_subvec;

    for (std::size_t i = 0; i < image.size(); i++) {
        if (i % RowLength == 0) {
            if (i != 0) result.push_back(last_subvec);
            last_subvec = {};
        } 
        last_subvec.push_back(image[i]);
    }
    result.push_back(last_subvec);
    return result;
}
