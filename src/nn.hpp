#pragma once
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <set>
#include "dataLoader.hpp"

using ActivationFunction = float(*)(float);
using LossFunction = float(*)(std::vector<float> const&, std::vector<float> const&);

inline float ReLu(float x) {
    return std::max(0.0f, x);
}

inline float ReLu_derivative(float x) {
    return x >= 0 ? 1 : 0;
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

inline float sigm(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float sigm_derivative(float x) {
    return sigm(x) * (1 - sigm(x));
}

struct WeightConfig {
    std::vector<std::vector<float>> weights;
    std::vector<float> biases; 
    ActivationFunction function;
    ActivationFunction derivative;
};

using DoubleVec = std::vector<std::vector<float>>;
using ConfigPart = std::vector<std::pair<DoubleVec, std::vector<float>>>;
ConfigPart load_weights_and_biases(std::string const& filepath);


struct FileConfig {
    std::vector<WeightConfig> layer_data;
    std::map<std::string, std::size_t> mapping;
    LossFunction loss;

    static FileConfig from_file(std::string const& filepath) {
        FileConfig result;
        static std::vector<ActivationFunction> preconfigured_funcs = {ReLu, ReLu, sigm,};
        static std::vector<ActivationFunction> preconfigured_der = {ReLu_derivative, ReLu_derivative, sigm_derivative};
        static std::map<std::string, std::size_t> preconfigured_mapping = {
            {"bee", 0},
            {"key", 1},
            {"carrot", 2},
            {"fish", 3},
            {"apple", 4},
        };

        auto loaded = load_weights_and_biases(filepath);
        for (std::size_t i = 0; i < loaded.size(); i++) {
            auto [weights, biases]  = loaded[i];
            result.layer_data.push_back({weights, biases, preconfigured_funcs[i], preconfigured_der[i]});
        }
        result.loss = MSE;
        result.mapping = preconfigured_mapping;
        return result;
    }
};



class Layer {
public:
    Layer(std::size_t in_size, std::size_t out_size, ActivationFunction activation, ActivationFunction derivative);
    Layer(WeightConfig const& config);

    std::vector<float> forward_pass(std::vector<float> const& previous) const;
    std::vector<float> sum_inputs(std::vector<float> const& previous) const;
    std::vector<float> const& get_node_gradient() const;

    std::vector<std::vector<float>> const& get_weights() const;
    std::vector<float> const& get_bias_weights() const;

    void update_gradient(float learning_rate);
    void calculate_out_layer_grad(
        std::vector<float> const& output_diff,
        std::vector<float> const& summed_out,
        std::vector<float> const& layer2_vals
    );
    void calculate_second_layer_grad(
        std::vector<float> const& layer0_vals,
        std::vector<float> const& sums,
        std::vector<float> const& out_wsum,
        std::vector<std::vector<float>> const& out_weights,
        std::vector<float> const& out_grad
    );

    void calculate_first_layer_grad(
        std::vector<float> const& image,
        std::vector<float> const& layer0_sum,
        std::vector<float> const& layer1_sum,
        std::vector<std::vector<float>> const& weights_l1,
        std::vector<float> const& layer1_grad
    );

    void reset_gradient(); 

private:
    static constexpr float INIT_MEAN = 0;
    static constexpr float INIT_DEVIATION = 0.123;

    std::vector<std::vector<float>> weights_;
    std::vector<float> biases_;
    std::vector<std::vector<float>> weight_gradient_;
    std::vector<float> bias_gradient_;
    std::vector<float> node_gradient_;
    
    std::size_t in_size_;
    std::size_t out_size_;
    ActivationFunction activation_;
    ActivationFunction activation_derivative_;
};

class NeuralNet {
public:
    struct Config {
        std::set<std::string> class_names;
        std::vector<std::pair<std::size_t, std::size_t>> layers_sizes;
        std::vector<ActivationFunction> functions;
        std::vector<ActivationFunction> derivatives;
        LossFunction loss_function;
    };

    NeuralNet(Config const& config);
    NeuralNet(FileConfig const& config);
    
    std::vector<float> forward_pass(std::vector<float> const& image) const;
    void learn(Dataset const& dataset, std::size_t L, float learning_rate);
    void dump_weights(std::string const& pathname) const;
    float calculate_total_cost(Dataset const& validation) const;
    void dump_predictions(Dataset const& datset, std::vector<std::size_t> const& indices) const;

    std::string get_result_name(std::size_t index) const;
    std::size_t get_result_index(std::string const& name) const;

private:
    LossFunction loss_function_;
    std::vector<Layer> layers_;
    std::map<std::string, std::size_t> name_to_index_;

    float calculate_cost(Record const& record) const;
    void calculate_gradients();
    void update_weigths(float lr, Record const& record);
    void reset_gradients();
    std::vector<float> get_output_differences(
        std::vector<float> const& predictions,
        std::string const& record_name
    ) const;
};
