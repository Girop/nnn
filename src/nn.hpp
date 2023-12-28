#pragma once
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <set>

using ActivationFunction = float(*)(float);
using OutFunction = std::vector<float>(*)(std::vector<float> const&);
using LossFunction = float(*)(std::vector<float> const&, std::vector<float> const&);

inline float ReLu(float x) {
    return std::max(0.0f, x); 
}

inline float MSE(std::vector<float> const& predictions, std::vector<float> const& valid) { 
    float sum = 0;
    for (unsigned long i = 0; i < predictions.size(); i++) {
        auto diff = (predictions[i] - valid[i]);
        sum += diff * diff;
    }
    sum /= predictions.size();
    return sum;
}

inline std::vector<float> SoftMax(std::vector<float> const& weights) {
    std::vector<float> result;
    float sum = 0.0f;
    for (auto value: weights) {
        sum += exp(value);
    }

    for (auto value: weights) {
        result.push_back(exp(value) / sum);
    }

    return result;
}

class Layer {
public:
    Layer(std::size_t in_size, std::size_t out_size, ActivationFunction activation);
    std::vector<float> forward_pass(std::vector<float> const& previous) const;

private:
    static constexpr float INIT_MEAN = 0.0;
    static constexpr float INIT_DEVIATION = 4.0;

    std::vector<std::vector<float>> weights_;
    std::vector<float> biases_;
    std::size_t in_size_;
    std::size_t out_size_;
    ActivationFunction activation_;
};

struct NeuralNetConfig {
    std::set<std::string> class_names;
    std::vector<std::pair<std::size_t, std::size_t>> layers_sizes;
    std::vector<ActivationFunction> functions;
    LossFunction loss_function;
};

class NeuralNet {
public:
    NeuralNet(NeuralNetConfig const& config);
    std::vector<float> forward_pass(std::vector<float> image) const;
    std::string get_result_name(std::size_t index) const;
    void backpropagation();
    // void calculate_loss();

private: 
    std::vector<Layer> layers_;
    std::map<std::string, std::size_t> name_to_index_;
    LossFunction loss_function_;
};
