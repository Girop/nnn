#pragma once
#include <vector>
#include <string>
#include <set>
#include <random>
#include <algorithm>

struct Record {
    std::string name;
    std::vector<float> image;
};

using Dataset = std::vector<Record>;
std::pair<Dataset, Dataset> split_dataset(Dataset const& data, float split_ratio);


class DataLoader {
public:
    DataLoader(std::string const& data_dir);
    void load(unsigned int batch_size = 1000, std::vector<std::string> const& names = {});
    Dataset const& get_data() const;

    static Dataset shuffle_data(Dataset const& orignal) {
        auto data = orignal;
        std::random_device rd;
        auto g = std::mt19937(rd());
        std::shuffle(data.begin(), data.end(), g);
        return data;
    }
    std::set<std::string> get_names() const;

private:
    std::vector<std::string> csv_filepaths_;
    Dataset loaded_data_;
    std::set<std::string> loaded_names_;
    Record parse_csv_line(std::string const& line);
};
