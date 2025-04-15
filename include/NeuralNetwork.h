//
// Created by MattFor on 12.04.2025.
//

// NeuralNetwork.h
// Defines a feedforward neural network composed of Layers.

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <span>
#include <vector>
#include <chrono>    // For timing in train
#include <cstddef>   // For std::size_t
#include <iostream>  // For std::cout (used in display_topology)
#include <stdexcept> // For std::invalid_argument

#include "Layer.h"

enum class NetworkComplexity
{
    SIMPLE, // Fewer layers/neurons, potentially faster training, prone to underfitting complex data
    MEDIUM, // Balanced approach, good starting point
    COMPLEX // More layers/neurons, potentially higher capacity, slower training, prone to overfitting small data
};

/**
 * @brief Represents a feedforward neural network.
 *
 * Composed of multiple Layers, this class handles the overall network structure,
 * forward propagation, backpropagation, training, and prediction.
 */
class NeuralNetwork
{
    std::size_t              input_size{};  ///< Number of input features expected by the network.
    std::size_t              output_size{}; ///< Number of output neurons in the final layer.
    std::vector<Layer>       layers;        ///< The sequence of layers composing the network.
    std::vector<std::size_t> layer_sizes;   ///< Stores the size of each layer.

public:
    NeuralNetwork() = default;

    /**
     * @brief Constructs a Neural Network based on specified layer sizes, activation functions, and weight initialization ranges.
     * @param layer_sizes Vector defining network structure. First element is input size,
     * subsequent elements are neuron counts for hidden and output layers. E.g., {784, 128, 10}.
     * @param hidden_activation Activation function type for all hidden layers.
     * @param output_activation Activation function type for the output layer.
     * @param weight_init_min Minimum value for random weight/bias initialization (passed to layers).
     * @param weight_init_max Maximum value for random weight/bias initialization (passed to layers).
     * @throws std::invalid_argument if layer_sizes has fewer than 2 elements or contains zeros inappropriately.
     */
    NeuralNetwork(
        const std::vector<std::size_t>& layer_sizes,
        const ActivationType            hidden_activation, // Added parameter
        const ActivationType            output_activation, // Added parameter
        double                          weight_init_min = -0.5,
        double                          weight_init_max = 0.5
    ) : layer_sizes(layer_sizes) // For use in saving to binary files.
    {
        // --- Validate Input ---
        if (layer_sizes.size() < 2)
        {
            throw std::invalid_argument(
                "NeuralNetwork Error: layer_sizes must contain at least an input size and an output size (2 elements minimum).");
        }

        if (layer_sizes[0] == 0)
        {
            throw std::invalid_argument(
                "NeuralNetwork Error: Input size (first element of layer_sizes) cannot be zero.");
        }

        for (size_t i = 1; i < layer_sizes.size(); i++)
        {
            if (layer_sizes[i] == 0)
            {
                throw std::invalid_argument(
                    "NeuralNetwork Error: Layer " + std::to_string(i) + " size cannot be zero.");
            }
        }

        // --- Store Basic Info ---
        this->input_size  = layer_sizes[0];
        this->output_size = layer_sizes.back();

        // --- Create Layers ---
        this->layers.reserve(layer_sizes.size() - 1); // Reserve space for actual layers

        // Loop starts from the first *hidden* or *output* layer (index 1 in layer_sizes)
        for (std::size_t i = 1; i < layer_sizes.size(); i++)
        {
            std::size_t num_neurons_in_layer  = layer_sizes[i];
            std::size_t num_inputs_per_neuron = layer_sizes[i - 1]; // Size of previous layer

            // Determine the activation type for the current layer being created.
            ActivationType current_activation = i == layer_sizes.size() - 1 ? output_activation : hidden_activation;

            // Call the Layer constructor, passing all necessary parameters:
            // size, inputs per neuron, activation type, weight init min/max
            this->layers.emplace_back(
                num_neurons_in_layer,
                num_inputs_per_neuron,
                current_activation, // Pass determined activation type
                weight_init_min,    // Pass weight init min
                weight_init_max     // Pass weight init max
            );
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
    void backpropagate(
        std::span<const double> network_input,
        std::span<const double> target,
        double                  learning_rate
    );

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
    void train(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets,
        std::size_t                             epochs,
        double                                  learning_rate
    );

    /**
     * @brief Gets the number of input features the network expects.
     * @return The input size.
     */
    [[nodiscard]] std::size_t get_input_size() const noexcept;

    /**
     * @brief Gets the number of output neurons in the network's final layer.
     * @return The output size.
     */
    [[nodiscard]] std::size_t get_output_size() const noexcept;

    /**
     * @brief Gets read-only access to the layers of the network.
     * @return A const reference to the vector of layers.
     */
    [[nodiscard]] const std::vector<Layer>& get_layers() const noexcept;

    /**
    * @brief Gets mutable access to the layers of the network.
    * @return A reference to the vector of layers.
    * @warning Modifying layers or their neurons directly can affect network state. Use with caution.
    */
    [[nodiscard]] std::vector<Layer>& get_layers_mut() noexcept;

    /**
     * @brief Gets the sizes of all layers in the network.
     * @return A const reference to the vector containing the size of each layer.
     */
    [[nodiscard]] const std::vector<std::size_t>& get_layer_sizes() const noexcept;

    /**
     * @brief Loads the network's weights and biases from a binary file.
     *
     * This method reads the network's topology and the weights and biases of each neuron
     * from a binary file. The topology in the file must exactly match the current
     * network's architecture for the weights to be loaded correctly.
     *
     * @param filename The name of the file to load the weights from.
     * @return True if the weights were loaded successfully, false otherwise (e.g., file not found,
     * topology mismatch, or error during reading).
     */
    bool load_weights(const std::string& filename);

    /**
     * @brief Saves the network's weights and biases to a binary file.
     *
     * This method serializes the network's topology (input size, output size, number of layers,
     * activation types, number of neurons per layer) and the weights and biases of each neuron
     * into a binary file. This allows for the trained state of the network to be saved and
     * loaded later.
     *
     * @param filename The name of the file to save the weights to.
     * @throws std::runtime_error if the file cannot be opened for writing.
     */
    void save_weights(const std::string& filename) const;

    /**
     * @brief Generates a plausible neural network layer topology based on input/output sizes,
     * training data amount, and desired complexity. Uses heuristics and rules of thumb.
     * @param num_input_features Size of the input layer.
     * @param num_output_neurons Size of the output layer.
     * @param num_training_samples Number of samples available for training (influences capacity).
     * @param complexity_level Hint for desired network complexity (SIMPLE, MEDIUM, COMPLEX).
     * @return A std::vector<std::size_t> representing the layer sizes [input, hidden1, ..., hiddenN, output].
     */
    static std::vector<std::size_t> generate_layer_topology(
        std::size_t       num_input_features,
        std::size_t       num_output_neurons,
        std::size_t       num_training_samples,
        NetworkComplexity complexity_level
    );
};

#endif // NEURALNETWORK_H
