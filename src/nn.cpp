#include "nn.hpp"
#include "utils.hpp"
#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>

Layer::Layer(
    std::size_t in_size,
    std::size_t out_size,
    ActivationFunction activation,
    ActivationFunction derivative
)
    : in_size_(in_size),
    out_size_(out_size),
    activation_(activation),
    activation_derivative_(derivative)
{
    static std::random_device rd;
    static auto gen = std::mt19937(rd());
    static auto normal = std::normal_distribution<float>(INIT_MEAN, INIT_DEVIATION);

    for (std::size_t _j = 0; _j < out_size; _j++) {
        std::vector<float> sub_weight, grad_subw;
        for (std::size_t _i = 0; _i < in_size; _i++) {
            sub_weight.push_back(normal(gen)); 
            grad_subw.push_back(0.0f);
        }
        weights_.push_back(sub_weight);
        biases_.push_back(normal(gen));
        weight_gradient_.push_back(grad_subw);
        bias_gradient_.push_back(0.0f);
        node_gradient_.push_back(0.0f);
    }
}

Layer::Layer(WeightConfig const& config):
    in_size_(config.weights[0].size()),
    out_size_(config.weights.size()),
    activation_(config.function),
    activation_derivative_(config.derivative) {

    node_gradient_.assign(out_size_, 0.0f);
    bias_gradient_.assign(out_size_, 0.0f);

    for (std::size_t _i = 0; _i < out_size_; _i++) {
        weight_gradient_.push_back(std::vector<float>(in_size_, 0.0f));
    }

    biases_ = config.biases;
    weights_ = config.weights;
}

std::vector<float> Layer::sum_inputs(std::vector<float> const& previous) const {
    std::vector<float> result;
    for (std::size_t j = 0; j < out_size_; j++) {
        auto sum = biases_[j];
        for (std::size_t i = 0; i < in_size_; i++) {
            sum += weights_[j][i] * previous[i];
        }
        result.push_back(sum);
    }
    return result;
}

std::vector<float> Layer::forward_pass(std::vector<float> const& previous) const {
    auto weighted_sums = sum_inputs(previous);
    for (auto& val: weighted_sums) {
        val = activation_(val);
    }
    return weighted_sums;
}

std::vector<float> const& Layer::get_node_gradient() const {
    return node_gradient_;
}

std::vector<std::vector<float>> const& Layer::get_weights() const {
    return weights_;
}

std::vector<float> const& Layer::get_bias_weights() const {
    return biases_;
}

void Layer::calculate_out_layer_grad(
        std::vector<float> const& output_diff,
        std::vector<float> const& summed_out,
        std::vector<float> const& incoming_values
) {
    for (unsigned int j = 0; j < 5; j++) {
        constexpr float constant = -(2.0f / 5.0f);
        float acti_derr = activation_derivative_(summed_out[j]);
        float common_factor = constant * output_diff[j] * acti_derr;

        for (unsigned int i = 0; i < 128; i++) {
            weight_gradient_[j][i] = common_factor * incoming_values[i];
        }

        bias_gradient_[j] = common_factor;
        node_gradient_[j] = constant * output_diff[j];
    }
}

void Layer::calculate_second_layer_grad(
    std::vector<float> const& layer0_vals,
    std::vector<float> const& sums,
    std::vector<float> const& out_wsum,
    std::vector<std::vector<float>> const& out_weights,
    std::vector<float> const& out_grad
) {
    for (unsigned int j = 0; j < 128; j++) {
        float next_layer_sum = 0.0f;
        for (unsigned int n = 0; n < 5; n++) {
            next_layer_sum += out_weights[n][j] * sigm_derivative(out_wsum[n]) * out_grad[n];
        }

        for (unsigned int i = 0; i < 256; i++) {
            weight_gradient_[j][i] = layer0_vals[i] * activation_derivative_(sums[j]) * next_layer_sum;
        }

        bias_gradient_[j] = activation_derivative_(sums[j]) * next_layer_sum;
        node_gradient_[j] = next_layer_sum;
    }
}

void Layer::calculate_first_layer_grad(
    std::vector<float> const& image,
    std::vector<float> const& layer0_sum,
    std::vector<float> const& layer1_sum,
    std::vector<std::vector<float>> const& weights_l1,
    std::vector<float> const& layer1_grad
) {
    for (unsigned int j = 0; j < 256; j++) {
        float next_layer_sum = 0.0f;
        for (unsigned int n = 0; n < 126; n++) {
            next_layer_sum += weights_l1[n][j] * activation_derivative_(layer1_sum[n]) * layer1_grad[n];
        }

        for (unsigned int i = 0; i < 65536; i++) {
            weight_gradient_[j][i] = image[i] * activation_derivative_(layer0_sum[j]) * next_layer_sum;
        }

        bias_gradient_[j] = activation_derivative_(layer0_sum[j]) * next_layer_sum;
    }
}

void Layer::update_gradient(float learning_rate) {
    for (unsigned int j = 0; j < weights_.size(); j++) {
        for (unsigned int i = 0; i < weights_[0].size(); i++) {
            weights_[j][i] -= learning_rate * weight_gradient_[j][i];
        }
        biases_[j] -= learning_rate * bias_gradient_[j];
    }
}

void Layer::reset_gradient() {
    for (auto& sub_vec: weight_gradient_) {
        sub_vec.assign(sub_vec.size(), 0.0f);
    }
    bias_gradient_.assign(bias_gradient_.size(), 0.0f);
    node_gradient_.assign(node_gradient_.size(), 0.0f);
}

NeuralNet::NeuralNet(Config const& config): loss_function_(config.loss_function) {
    for (std::size_t i = 0; i < config.layers_sizes.size(); i++) {
        auto sizes = config.layers_sizes[i];
        layers_.push_back({
            sizes.first,
            sizes.second,
            config.functions[i],
            config.derivatives[i]
        });
    }
    
    std::size_t idx = 0;
    for (auto const& name: config.class_names) {
        name_to_index_[name] = idx++;
    }
}

NeuralNet::NeuralNet(FileConfig const& config):
    loss_function_(config.loss),
    name_to_index_(config.mapping) {

    for (auto const& conf: config.layer_data) {
        layers_.push_back({conf});
    }
    
}

std::vector<float> NeuralNet::forward_pass(std::vector<float> const& image) const {
    auto zs = image;
    for (auto const& layer: layers_) {
        zs = layer.forward_pass(zs);
    }
    return zs;
}


std::string NeuralNet::get_result_name(std::size_t index) const {
    for (auto const& pair: name_to_index_) {
        if (pair.second == index) {
            return pair.first;
        }
    }
    std::cerr << "Unkown index: " << index << std::endl;
    exit(1);
}

std::size_t NeuralNet::get_result_index(std::string const& name) const {
    auto result = name_to_index_.find(name);
    if (result != name_to_index_.end()) {
        return result->second;
    }

    std::cerr << "Unkown name: " << name << std::endl;
    exit(1);
}

void NeuralNet::learn(Dataset const& dataset, std::size_t L, float learning_rate) {
    for (unsigned int n = 0; n <= L; n++) {
        displayProgressBar(n + 1, L + 1);
        std::size_t record_index = L % dataset.size();
        update_weigths(learning_rate, dataset[record_index]);
        reset_gradients();
        iteration_loss_.push_back(calculate_cost(dataset[record_index]));
    } 
}

float NeuralNet::calculate_cost(Record const& record) const {
    auto prediction = forward_pass(record.image);
    auto desired = std::vector<float>(prediction.size(), 0.0f);  
    auto valid_class_idx = name_to_index_.find(record.name)->second;
    desired[valid_class_idx] = 1.0f;
    return loss_function_(prediction, desired);
}

std::vector<float> NeuralNet::get_output_differences(
    std::vector<float> const& predictions,
    std::string const& record_name
) const {
    auto result = std::vector<float>(predictions.size(), 0.0f);
    result[get_result_index(record_name)] = 1.0f;
    for (unsigned int i = 0; i < result.size(); i++) {
        result[i] -=  predictions[i];
    }
    return result;
}

void NeuralNet::update_weigths(float lr, Record const& record) {
    auto layer0_values = layers_[0].forward_pass(record.image);
    auto layer1_values = layers_[1].forward_pass(layer0_values);
    auto predictions = layers_[2].forward_pass(layer1_values);

    auto wsum0 = layers_[0].sum_inputs(record.image);
    auto wsum1 = layers_[1].sum_inputs(layer0_values);
    auto out_wsum = layers_[2].sum_inputs(layer1_values);

    auto const& out_node_grad = layers_[2].get_node_gradient();
    auto const& out_node_weights = layers_[2].get_weights();
    auto const& layer1_grad = layers_[1].get_node_gradient(); 
    auto const& layer1_weigthts = layers_[1].get_weights();

    auto output_diff = get_output_differences(predictions, record.name);
    
    layers_[2].calculate_out_layer_grad(output_diff, out_wsum, layer1_values);
    layers_[1].calculate_second_layer_grad(layer0_values, wsum1, out_wsum, out_node_weights, out_node_grad);
    layers_[0].calculate_first_layer_grad(record.image, wsum0, wsum1, layer1_weigthts, layer1_grad);

    for (auto& layer: layers_) {
        layer.update_gradient(lr);
    }
}

void NeuralNet::reset_gradients() {
    for (auto& layer: layers_) {
        layer.reset_gradient();
    }
}

float NeuralNet::calculate_total_cost(Dataset const& test) const {
    float result = 0.0f;
    for (auto& record: test) {
        result += calculate_cost(record);
    }
    return result;
}

void NeuralNet::dump_weights(std::string const& pathname) const {
    std::stringstream output;
    for (std::size_t i = 0; i < layers_.size(); i++) {
        output << "Layer " << i << '\n';
        output << "Weights:\n";
        std::size_t node_index = 0;
        for (auto const& node_w: layers_[i].get_weights()) {
            output << "Node " << node_index++ << ": ";
            for (auto val: node_w) {
                output << val << ' ';
            }
            output << '\n';
        }
        
        output << "Bias weights:";
        for (auto b_i: layers_[i].get_bias_weights()) { 
            output << b_i << ' ';
        }
        output << '\n';
    }

    auto output_file = std::ofstream(pathname, std::ios::out | std::ios::trunc);
    if (!output_file.is_open()) {
        std::cerr << "Weight dump failed!\n";
        exit(1);
    }
    output_file << output.str();
    std::cout << "Weights saved\n";
}

void NeuralNet::dump_predictions(std::string const& dumpdir,std::string const& parentdir, Dataset const& datset) const {   
    std::vector<std::string> descriptions;

    for (std::size_t index = 0; index < datset.size(); index++) {
        auto image = datset[index].image;
        auto path = dumpdir + std::to_string(index) + ".ppm";
        auto img = unflatten_image<256>(image);
        save_to_ppm(path, img);

        auto image_description = "File: " + path + ", Class: " + datset[index].name + '\n';
        auto pred = forward_pass(image);
        for (std::size_t i = 0; i < pred.size(); i++) {
            image_description += get_result_name(i) + ":" + std::to_string(pred[i]) + "\n";
        }
        descriptions.push_back(image_description);
    }
    
    std::string description_string = "Predictions for files in this directory\n";
    for (auto& desc: descriptions) {
        description_string += "\n" + desc + "\n";
    }

    auto description_file = std::ofstream(parentdir + "predictions.txt", std::ios::out | std::ios::trunc);
    if (!description_file.is_open()) {
        std::cerr << "Failed to save predictions description file" << std::endl;
        exit(1);
    }
    description_file << description_string;
    std::cout << "Predictions saved\n";
}

void NeuralNet::dump_iterations(std::string const& dumppath, Dataset const& dataset) const {
    auto file = std::ofstream(dumppath, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Coulnd save iteration info!\n";
        exit(1);
    }
    
    file << "Iteration count: " << iteration_loss_.size() << '\n'
    << "Total cost: " << calculate_total_cost(dataset) << '\n';

    for (std::size_t i = 0; i < iteration_loss_.size(); i++) {
        file << iteration_loss_[i] << ' ';
    }
}

void NeuralNet::dump_statistics(std::string const& dumpdir, Dataset const& dataset) const {
    auto images_dir = dumpdir + "/images/";
    try {
        std::filesystem::create_directory(dumpdir);
        std::filesystem::create_directory(images_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
    }

    dump_weights(dumpdir + "/weights.txt");
    dump_iterations(dumpdir + "/iterations.txt", dataset);
    dump_predictions(images_dir, dumpdir, dataset);
}


static std::vector<float> parse_line_values(std::string const& line) {
    auto colon_pos = line.find(':');
    std::vector<float> result;
    if (colon_pos != std::string::npos) {
        auto iss = std::istringstream(line.substr(colon_pos + 1));
        float value;
        while (iss >> value) 
            result.push_back(value);
    } else {
        std::cerr << "Invalid line\n";
        exit(1);
    }
    return result;
}

using DoubleVec = std::vector<std::vector<float>>;
using ConfigPart = std::vector<std::pair<DoubleVec, std::vector<float>>>;

ConfigPart load_weights_and_biases(std::string const& filepath) {
    ConfigPart result;
    auto file = std::ifstream(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to load data\n";
        exit(1);
    }

    std::string line;
    std::vector<std::vector<float>> last_weights;
    std::vector<float> last_biases; 

    std::size_t layer_index;
    while (std::getline(file, line)) {
        if (line.find("Layer") != std::string::npos) {
            last_weights = {};
            last_biases = {};
            layer_index++;
            continue;
        }

        if (line.find("Node") != std::string::npos) {
            last_weights.push_back(parse_line_values(line));
        } else if (line.find("Bias") != std::string::npos) {
            last_biases = parse_line_values(line);
            result.push_back({last_weights, last_biases});
        }
    }
    return result;
}
