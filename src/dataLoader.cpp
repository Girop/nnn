#include "dataLoader.hpp"
#include <sstream>
#include <fstream>
#include <iostream>
#include <filesystem>


DataLoader::DataLoader(std::string const& data_dir) {
    using namespace std::filesystem;
    auto data_path = path{data_dir};
     
    try {
        if (!is_directory(data_dir)) {
            std::cerr << "Given path is not dir!" << '\n';
            return;
        }

        for (auto const& entry: directory_iterator(data_dir)) {
            if (entry.is_directory()) continue;
            csv_filepaths_.push_back(entry.path());
        }

    } catch (filesystem_error const& er) {
        std::cerr << "Error acessing file: " << er.what() << '\n';
    }
}

void DataLoader::load(unsigned int batch_size, std::vector<std::string> const& names) {
    bool name_filer_active = names.size() > 0;
    for (auto const& path: csv_filepaths_) {
        auto file = std::fstream(path);
        if (!file.is_open()) {
            std::cout << "Failed opening file: "  << path << '\n';
            continue;
        }
        std::cout << "Loading: " << path << '\n';
        unsigned int line_count = 0;
        std::string line;
        auto should_load = [&](std::string const& entry_name){
            return std::find(names.begin(), names.end(), entry_name) != names.end();
        };

        while (std::getline(file, line)) {
            if (line_count++ == 0) continue;
            auto entry = parse_csv_line(line);
            if (name_filer_active && !should_load(entry.name)) break;
            loaded_data_.push_back(entry);
            if (line_count > batch_size) break;
        }
    } 
}

Record DataLoader::parse_csv_line(std::string const& line) {
    static constexpr char delimiter = ',';
    Record result;
    auto token_stream = std::istringstream(line);
    std::string token;
    unsigned int token_counter = 0;
    while (std::getline(token_stream, token, delimiter)) {
        if (++token_counter == 1) {
            result.name = token;
            loaded_names_.insert(token);
        } else {
            result.image.push_back(std::stof(token));
        }
    }
    return result;
}

Dataset const& DataLoader::get_data() const {
    return loaded_data_;
}


std::set<std::string> DataLoader::get_names() const{
    return loaded_names_;
}


std::pair<Dataset, Dataset> split_dataset(Dataset const& data, float split_ratio) {
    auto split_point = data.begin() + data.size() * split_ratio;
    return {
        {data.begin(), split_point},
        {split_point, data.end()}
    };
}
