#include "nn.hpp"
#include <random>


Layer::Layer(
    std::size_t in_size,
    std::size_t out_size,
    ActivationFunction activation
): in_size_(in_size), out_size_(out_size), activation_(activation) {
    std::random_device rd;
    auto gen = std::mt19937{rd()};
    auto normal = std::normal_distribution<float>{INIT_MEAN, INIT_DEVIATION};

    for (std::size_t _i = 0; _i < out_size; _i++) {
        std::vector<float> sub_weight;
        for (std::size_t _j = 0; _j < in_size; _j++) {
            sub_weight.push_back(normal(gen));
        }
        weights_.push_back(sub_weight);
        biases_.push_back(normal(gen));
    }
}

std::vector<float> Layer::forward_pass(std::vector<float> const& previous) const {
    std::vector<float> result;
    for (std::size_t neuron_idx = 0; neuron_idx < out_size_ ; neuron_idx++) {
        auto sum = biases_[neuron_idx];
        for (std::size_t idx = 0; idx < in_size_; idx++) {
            sum = weights_[neuron_idx][idx] * previous[idx];
        }
        result.push_back(activation_(sum));
    }
    return result;
}

NeuralNet::NeuralNet(NeuralNetConfig const& config): loss_function_(config.loss_function) {
    for (std::size_t i = 0; i < config.layers_sizes.size(); i++) {
        auto sizes = config.layers_sizes[i];
        layers_.push_back({sizes.first, sizes.second, config.functions[i]});
    }
    
    std::size_t idx = 0;
    for (auto const& name: config.class_names) {
        name_to_index_[name] = idx++;
    }
}

std::vector<float> NeuralNet::forward_pass(std::vector<float> image) const {
    auto weights = image;
    for (auto const& layer: layers_) {
        weights = layer.forward_pass(weights);
    }
    return SoftMax(weights);
}


std::string NeuralNet::get_result_name(std::size_t index) const {
    for (auto const& pair: name_to_index_) {
        if (pair.second == index) {
            return pair.first;
        }
    }
    return "";
}

void NeuralNet::backpropagation() {

}
