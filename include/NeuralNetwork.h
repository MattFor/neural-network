// NeuralNetwork.h
// Defines a feedforward neural network composed of Layers.

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <span>
#include <vector>
#include <cstddef> // For std::size_t
#include <stdexcept> // For std::invalid_argument
#include <string>    // For std::to_string, std::string
#include <iostream>  // For std::cout (used in display_topology)
#include <iomanip>   // For std::setw (used in display_topology)
#include <limits>    // For std::numeric_limits
#include <cmath>     // For std::log, std::max, std::min
#include <numeric>   // For std::accumulate (potentially useful)
#include <chrono>    // For timing in train
#include <cstdio>    // For std::printf, std::fflush in train
#include <typeinfo>  // For typeid in display_topology

#include "Layer.h" // Include the definition of the Layer class

/**
 * @brief Represents a feedforward neural network.
 *
 * Composed of multiple Layers, this class handles the overall network structure,
 * forward propagation, backpropagation, training, and prediction.
 */
class NeuralNetwork
{
private:
    std::size_t        input_size;  ///< Number of input features expected by the network.
    std::size_t        output_size; ///< Number of output neurons in the final layer.
    std::vector<Layer> layers;      ///< The sequence of layers composing the network.

public:
    /**
     * @brief Constructs a Neural Network based on specified layer sizes.
     * @param layer_sizes A vector where each element represents the number of neurons
     * in a layer. The first element is the input size (not a layer),
     * followed by the sizes of hidden layers, and ending with the
     * output layer size. Must contain at least 2 elements (input size, output size).
     * @param weight_init_min Minimum value for random weight/bias initialization (passed to layers).
     * @param weight_init_max Maximum value for random weight/bias initialization (passed to layers).
     * @throws std::invalid_argument if layer_sizes has fewer than 2 elements.
     */
    explicit NeuralNetwork(const std::vector<std::size_t>& layer_sizes,
                           double                          weight_init_min = -0.5,
                           double                          weight_init_max = 0.5)
    {
        if (layer_sizes.size() < 2)
        {
            throw std::invalid_argument(
                "NeuralNetwork Error: layer_sizes must contain at least an input size and an output size (2 elements minimum).");
        }
        input_size  = layer_sizes[0];
        output_size = layer_sizes.back();

        // Create hidden and output layers (start from index 1 of layer_sizes)
        layers.reserve(layer_sizes.size() - 1); // Reserve space
        for (std::size_t i = 1; i < layer_sizes.size(); ++i)
        {
            std::size_t num_neurons_in_layer  = layer_sizes[i];
            std::size_t num_inputs_per_neuron = layer_sizes[i - 1]; // Size of the previous layer (or input size)

            // Call the updated Layer constructor
            layers.emplace_back(num_neurons_in_layer, num_inputs_per_neuron, weight_init_min, weight_init_max);
        }
    }

    /**
     * @brief Displays the network's topology and some sample parameters.
     */
    void display_topology() const;

    /**
     * @brief Performs a forward pass through the entire network for a single input sample.
     * @param input A view of the input features for one sample. Size must match network's input_size.
     * @return A vector containing the output activations of the final layer.
     * @throws std::runtime_error If input size mismatch occurs.
     * @note This method is non-const because it calls non-const Layer::feed_forward.
     */
    [[nodiscard]] std::vector<double> feed_forward(std::span<const double> input);

    /**
     * @brief Performs a forward pass for a batch of input samples.
     * @param inputs A vector where each element is an input feature vector for one sample.
     * @return A vector where each element is the output vector for the corresponding input sample.
     * @throws std::runtime_error If any input sample size mismatch occurs.
     * @note This method is non-const.
     */
    [[nodiscard]] std::vector<std::vector<double>> feed_forward(const std::vector<std::vector<double>>& inputs);

    /**
     * @brief Performs backpropagation for a single input sample to update network weights.
     *
     * Assumes that feed_forward(network_input) was called immediately prior,
     * ensuring neuron and layer outputs are cached and up-to-date.
     *
     * @param network_input The original input vector for this training sample. Size must match network's input_size.
     * @param target The target output vector for this training sample. Size must match network's output_size.
     * @param learning_rate The learning rate for weight updates.
     * @throws std::runtime_error If input or target size mismatch occurs.
     */
    void backpropagate(std::span<const double> network_input,
                       std::span<const double> target,
                       double                  learning_rate);

    /**
     * @brief Predicts the output for a single input sample. (Semantic alias for feed_forward).
     * @param input A view of the input features for one sample.
     * @return A vector containing the output activations of the final layer.
     * @note This method is non-const.
     */
    [[nodiscard]] std::vector<double> predict(std::span<const double> input);

    /**
     * @brief Predicts the outputs for a batch of input samples. (Semantic alias for feed_forward).
     * @param inputs A vector where each element is an input feature vector for one sample.
     * @return A vector where each element is the output vector for the corresponding input sample.
     * @note This method is non-const.
     */
    [[nodiscard]] std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& inputs);

    /**
     * @brief Computes the average Binary Cross-Entropy loss between network output and target.
     * @param output The output vector from the network (predictions).
     * @param target The target vector.
     * @return The computed average loss value.
     * @throws std::runtime_error if output and target sizes mismatch.
     * @note Uses a small epsilon for numerical stability.
     */
    static double compute_loss(const std::vector<double>& output, const std::vector<double>& target);

    /**
     * @brief Trains the network on a dataset for a specified number of epochs.
     * @param inputs A vector of input feature vectors (the dataset).
     * @param targets A vector of corresponding target output vectors.
     * @param epochs The number of times to iterate over the entire dataset.
     * @param learning_rate The learning rate for weight updates during backpropagation.
     * @throws std::runtime_error If the number of input samples doesn't match the number of target samples,
     * or if individual sample sizes don't match network input/output sizes.
     */
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               std::size_t                             epochs,
               double                                  learning_rate);

    /**
     * @brief Gets the number of input features the network expects.
     * @return The input size.
     */
    [[nodiscard]] std::size_t get_input_size() const noexcept { return input_size; }

    /**
     * @brief Gets the number of output neurons in the network's final layer.
     * @return The output size.
     */
    [[nodiscard]] std::size_t get_output_size() const noexcept { return output_size; }

    /**
     * @brief Gets read-only access to the layers of the network.
     * @return A const reference to the vector of layers.
     */
    [[nodiscard]] const std::vector<Layer>& get_layers() const noexcept { return layers; }

    /**
    * @brief Gets mutable access to the layers of the network.
    * @return A reference to the vector of layers.
    * @warning Modifying layers or their neurons directly can affect network state. Use with caution.
    */
    [[nodiscard]] std::vector<Layer>& get_layers_mut() noexcept { return layers; } // Added if needed
};

#endif // NEURALNETWORK_H
