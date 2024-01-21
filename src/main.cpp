#include "dataLoader.hpp"
#include "nn.hpp"
#include "utils.hpp"
#include <iostream>

GlobalConfig global_config {
    200,
    200,
    {"bee", "carrot", "key"},
    1.0f,
    RunMode::Learn,
    "result"
};

void learn_nn() {
    auto loader = DataLoader("data");
    std::cout << "Loading dataset...\n";
    loader.load(global_config.image_count_per_category, global_config.categories);
    auto data = loader.get_data();
    auto [training, test_datset] = split_dataset(data, 0.7f);
    training = DataLoader::shuffle_data(training);

    std::cout << "Loaded " <<  data.size() << " images\n";
    auto categories = loader.get_names(); 

    auto config = NeuralNet::Config {
        categories,
        {
            {256 * 256, 256},
            {256, 128},
            {128, 5},
        },
        {ReLu, ReLu, sigm},
        {ReLu_derivative, ReLu_derivative, sigm_derivative},
        MSE,
    };

    std::cout << "Creating network\n";
    auto net = NeuralNet(config);
    float start_loss = net.calculate_total_cost(test_datset);

    std::cout << "Learning...\n";
    net.learn(training, global_config.L_learn_iterations, global_config.learn_rate);
    std::cout << "\nFinished learning\n";
    float end_loss = net.calculate_total_cost(test_datset);

    std::cout 
        << "Cost before learning: " << start_loss 
        << "\nCost after learning: " << end_loss 
        << std::endl;

    net.dump_statistics(global_config.result_dirname, test_datset);
}

void load_nn() {
    auto nn_config = FileConfig::from_file("result/weights.txt");
    auto nn = NeuralNet(nn_config);

    auto loader = DataLoader("data");
    loader.load(global_config.image_count_per_category, global_config.categories);
    auto const& data = loader.get_data();
    auto [training, test_datset] = split_dataset(data, 0.7f);
    training = DataLoader::shuffle_data(training);
    std::cout << "Loaded " <<  data.size() << " images\n";
}

int main() { 
    switch (global_config.mode) {
        case RunMode::Learn:
            learn_nn();
            break;
        case RunMode::Load:
            load_nn();
            break;
    }

    return 0;
}
