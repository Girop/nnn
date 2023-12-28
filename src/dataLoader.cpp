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

        for (auto const& entry: directory_iterator{data_dir}) {
            csv_filepaths_.push_back(entry.path());
        }

    } catch (filesystem_error const& er) {
        std::cerr << "Error acessing file: " << er.what() << '\n';
    }
}

void DataLoader::load_all(unsigned int batch_size) {
    for (auto const& path: csv_filepaths_) {
        auto file = std::fstream{path};
        if (!file.is_open()) {
            std::cout << "Failed opening file: "  << path << '\n';
            continue;
        }
        unsigned int line_count = 0;
        std::string line;
        while (std::getline(file, line)) {
            if (++line_count == 1) continue;
            if (line_count > batch_size) break;
            loaded_data_.push_back(parse_csv_line(line));
        }
    } 
}

static constexpr char delimiter = ',';

DataLoader::Record DataLoader::parse_csv_line(std::string const& line) {
    Record result;

    auto token_stream = std::istringstream{line};
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

DataLoader::Data const& DataLoader::get_data() const {
    return loaded_data_;
}

std::set<std::string> DataLoader::get_names() const{
    return loaded_names_;
}
