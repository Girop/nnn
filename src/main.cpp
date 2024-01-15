#include "dataLoader.hpp"
#include "nn.hpp"
#include <iostream>


enum class RunMode {
    Learn,
    Load,
};

constexpr unsigned int L_learn_iterations = 100;
constexpr float learn_rate = 1.0f / 200.0f;
constexpr unsigned int category_image_count = 100;
std::vector<std::string> filters = {"apple", "fish"};
std::string weight_save_path = "weights/weights.txt";

auto mode = RunMode::Learn;
std::string weight_load_path = "weights/weights.txt";

// TODO 
// Get correct parameters
//
// Running experiments
// Fix report
// Experiments report


void learn_nn() {
    auto loader = DataLoader("data");
    std::cout << "Loading dataset...\n";
    loader.load(category_image_count, filters);
    auto const& data = loader.get_shuffled_data();
    std::cout << "Loaded " <<  data.size() << " images\n";
    auto categories = loader.get_names(); 
    auto [training, validation] = split_dataset(data, 0.7f);

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
    float start_loss = net.calculate_total_cost(validation);

    std::cout << "Learning...\n";
    net.learn(training, L_learn_iterations, learn_rate);
    std::cout << "\nFinished learning\n";
    float end_loss = net.calculate_total_cost(validation);
    std::cout 
        << "Loss before learning: " << start_loss 
        << "\nLoss after learning: " << end_loss 
        << std::endl;
    net.dump_weights(weight_save_path);
    std::cout << "Weights saved\n";
    net.dump_predictions(validation, {1,2,3,4,5});
    std::cout << "Predictions saved\n";
}

void load_nn() {
    auto config = FileConfig::from_file(weight_load_path);
    auto nn = NeuralNet(config);

    auto loader = DataLoader("data");
    loader.load(category_image_count, filters);
    auto const& data = loader.get_shuffled_data();
    std::cout << "Loaded " <<  data.size() << " images\n";
    auto [training, validation] = split_dataset(data, 0.7f);

    auto pred = nn.forward_pass(training[0].image);
    auto cost = nn.calculate_total_cost(validation);
    std::cout << "Total cost: " << cost << '\n';
}

int main() {
    switch (mode) {
        case RunMode::Learn:
            learn_nn();
            break;
        case RunMode::Load:
            load_nn();
            break;
    }

    return 0;
}
