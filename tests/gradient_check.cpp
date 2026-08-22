#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

#include "NeuralNetwork.h"

namespace
{
    // Every trainable parameter of a network so a perturbed run can be reset exactly.
    struct Parameters
    {
        std::vector<std::vector<std::vector<double>>> weights;
        std::vector<std::vector<double>>              biases;
    };

    Parameters capture(const NeuralNetwork& network)
    {
        Parameters snapshot;

        for (const Layer& layer : network.get_layers())
        {
            std::vector<std::vector<double>> layer_weights;
            std::vector<double>              layer_biases;

            for (const Neuron& neuron : layer.get_neurons())
            {
                layer_weights.push_back(neuron.get_weights());
                layer_biases.push_back(neuron.get_bias());
            }

            snapshot.weights.push_back(std::move(layer_weights));
            snapshot.biases.push_back(std::move(layer_biases));
        }

        return snapshot;
    }

    void restore(NeuralNetwork& network, const Parameters& snapshot)
    {
        for (std::size_t l = 0; l < snapshot.weights.size(); l++)
        {
            for (std::size_t n = 0; n < snapshot.weights[l].size(); n++)
            {
                Neuron& neuron           = network.get_layers_mut()[l].get_neurons_mut()[n];
                neuron.get_weights_mut() = snapshot.weights[l][n];
                neuron.set_bias(snapshot.biases[l][n]);
            }
        }
    }

    double loss_at(NeuralNetwork& network, const std::vector<double>& input, const std::vector<double>& target)
    {
        return NeuralNetwork::compute_loss(network.feed_forward(std::span(input)), target);
    }

    struct Result
    {
        double hidden_error;
        double output_error;
    };

    Result check(const ActivationType hidden_activation, const int trials)
    {
        const std::vector<std::size_t> topology{
            4,
            5,
            3,
            2
        };
        const std::vector input{
            0.31,
            -0.77,
            0.52,
            0.11
        };
        const std::vector target{
            1.0,
            0.0
        };

        Result result{
            .hidden_error = 0.0,
            .output_error = 0.0
        };

        for (int trial = 0; trial < trials; trial++)
        {
            constexpr double learning_rate = 1.0;
            // Weights are randomised per construction, so each trial probes a new point.
            NeuralNetwork    network(topology, hidden_activation, ActivationType::SIGMOID);
            const Parameters before = capture(network);

            (void)network.feed_forward(std::span(input));
            network.backpropagate(input, target, learning_rate);

            const auto [weights, biases] = capture(network);
            restore(network, before);

            for (std::size_t l = 0; l < before.weights.size(); l++)
            {
                for (std::size_t n = 0; n < before.weights[l].size(); n++)
                {
                    for (std::size_t w = 0; w < before.weights[l][n].size(); w++)
                    {
                        constexpr double step     = 1e-5;
                        const double     analytic = ( before.weights[l][n][w] - weights[l][n][w] ) / learning_rate;

                        double&      weight   = network.get_layers_mut()[l].get_neurons_mut()[n].get_weights_mut()[w];
                        const double original = weight;

                        weight                  = original + step;
                        const double loss_plus  = loss_at(network, input, target);
                        weight                  = original - step;
                        const double loss_minus = loss_at(network, input, target);
                        weight                  = original;

                        const double numeric = ( loss_plus - loss_minus ) / ( 2.0 * step ) * static_cast<double>(network.get_output_size());

                        const double scale = std::max({
                            1e-3,
                            std::abs(numeric),
                            std::abs(analytic)
                        });

                        const double error = std::abs(numeric - analytic) / scale;

                        double& worst = l + 1 == before.weights.size() ? result.output_error : result.hidden_error;
                        worst         = std::max(worst, error);
                    }
                }
            }
        }

        return result;
    }
}

int main()
{
    constexpr int trials = 25;

    constexpr struct
    {
        const char*    name;
        ActivationType activation;
    } cases[] = {
        {
            .name = "ReLU",
            .activation = ActivationType::RELU
        },
        {
            .name = "LeakyReLU",
            .activation = ActivationType::LEAKY_RELU
        },
        {
            .name = "Tanh",
            .activation = ActivationType::TANH
        },
        {
            .name = "Sigmoid",
            .activation = ActivationType::SIGMOID
        },
    };

    int failures = 0;

    std::printf("Gradient check (topology 4-5-3-2, sigmoid output, %d trials per case)\n", trials);

    for (const auto& [name, activation] : cases)
    {
        constexpr double tolerance                    = 1e-4;
        const auto       [hidden_error, output_error] = check(activation, trials);
        const bool       passed                       = hidden_error < tolerance && output_error < tolerance;

        failures += !passed;

        std::printf("- hidden=%-10s max relative error: hidden layers %.3e | output layer %.3e  [%s]\n", name, hidden_error, output_error, passed ? "PASS" : "FAIL");
    }

    std::printf("%s\n", failures == 0 ? "All gradient checks passed." : "GRADIENT CHECK FAILED");

    return failures == 0 ? 0 : 1;
}
