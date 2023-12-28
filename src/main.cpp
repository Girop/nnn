#include "dataLoader.hpp"
#include "nn.hpp"


int main() {
    auto loader = DataLoader{"data"};
    loader.load_all();
    DataLoader::Data const& data = loader.get_data();
    auto categories =  loader.get_names();

    NeuralNetConfig config {
        categories, 
        {
            {256 * 256, 256},
            {256, 128},
            {128, 4},
        },
        {ReLu, ReLu, ReLu},
        MSE,
    };

    auto net = NeuralNet{config};
    net.forward_pass(data[0].image);
    return 0;
}
