#pragma once
#include <vector>
#include <string>
#include <set>


class DataLoader {
public:
    struct Record {
        std::string name;
        std::vector<float> image;
    };
    using Data = std::vector<Record>;

    DataLoader(std::string const& data_dir);
    void load_all(unsigned int batch_size = 1000);
    Data const& get_data() const;
    std::set<std::string> get_names() const;

private:
    std::vector<std::string> csv_filepaths_{};
    Data loaded_data_{};
    std::set<std::string> loaded_names_{};
    Record parse_csv_line(std::string const& line);
};
