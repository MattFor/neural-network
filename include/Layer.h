//
// Created by MattFor on 12.04.2025.
//

// Layer.h
// Represents a layer of neurons in a neural network.

#ifndef LAYER_H
#define LAYER_H

#include <span>    // For std::span
#include <vector>
#include <cstddef> // For std::size_t

#include "Neuron.h"

/**
 * @brief Represents a single layer in a feedforward neural network.
 *
 * Manages a collection of Neuron objects, handles feed-forward calculation
 * for the layer, and orchestrates backpropagation through its neurons.
 */
class Layer
{
    std::vector<Neuron> neurons; ///< Neurons composing this layer.
    std::vector<double> outputs; ///< Cached output activations from the last feed_forward pass.

    ActivationType layer_activation_type;

public:
    /**
     * @brief Constructs a Layer.
     * @param num_neurons The number of neurons in this layer.
     * @param num_inputs_per_neuron The number of inputs connecting to EACH neuron
     * (i.e., the number of neurons in the previous layer).
     * @param activation Which activation function should we use for our neurons.
     * @param weight_init_min Minimum value for random weight/bias initialization (passed to neurons).
     * @param weight_init_max Maximum value for random weight/bias initialization (passed to neurons).
     */
    Layer(
        std::size_t    num_neurons,
        std::size_t    num_inputs_per_neuron,
        ActivationType activation,
        double         weight_init_min = -0.5,
        double         weight_init_max = 0.5
    );

    /**
     * @brief Performs a feed-forward pass for this layer.
     *
     * Calculates the output of each neuron based on the inputs and updates
     * the internal output cache.
     *
     * @param inputs A view of the output vector from the previous layer (or the network's input).
     * @return A reference to the internally cached vector of output activations for this layer.
     * @throws std::invalid_argument if the size of inputs does not match num_inputs_per_neuron.
     * @note This method is not const because it calls the non-const Neuron::activate.
     */
    [[nodiscard]]
    const std::vector<double>& feed_forward(std::span<const double> inputs);

    /**
     * @brief Performs backpropagation for this layer.
     *
     * Computes gradients for all neurons in the layer based on the error signals
     * (either target values or weighted gradients from the next layer) and
     * updates their weights and biases. Optionally applies gradient clipping.
     *
     * @param error_signals For the output layer, this should be the target values corresponding
     * to each neuron in this layer. For hidden layers, this parameter's
     * content is typically ignored, as gradients are derived from the next layer.
     * Its size must match the number of neurons if this is the output layer.
     * @param prev_layer_outputs A view of the activation outputs from the PREVIOUS layer
     * (which served as inputs during the feed_forward pass).
     * Its size must match num_inputs_per_neuron.
     * @param learning_rate The learning rate for weight updates.
     * @param next_layer_ptr A pointer to the next layer (in the forward direction).
     * If `nullptr`, this layer is treated as the output layer.
     * If not `nullptr`, it's treated as a hidden layer, and gradients
     * are calculated based on the weights and gradients of the `next_layer`.
     * @param gradient_clip_threshold Apply clipping to the gradient to prevent data going out of scope.
     * @throws std::out_of_range If treated as output layer and `error_signals.size()` doesn't match neuron count.
     * @throws std::invalid_argument If `prev_layer_outputs.size()` doesn't match `num_inputs_per_neuron`.
     * @note Assumes Neuron class has `set_gradient` method if gradient clipping is enabled internally.
     */
    void backpropagate(
        std::span<const double> error_signals,
        std::span<const double> prev_layer_outputs,
        double                  learning_rate,
        const Layer*            next_layer_ptr,
        double                  gradient_clip_threshold // Default is 0.0 (no clipping)
    );

    // --- Getters ---

    /**
     * @brief Gets the cached output activations from the last feed-forward pass.
     * @return A constant reference to the vector of output values.
     */
    [[nodiscard]] const std::vector<double>& get_outputs() const noexcept;

    /**
     * @brief Provides access to the neurons in this layer.
     * @return A constant reference to the vector of Neuron objects.
     */
    [[nodiscard]] const std::vector<Neuron>& get_neurons() const noexcept;

    /**
    * @brief Provides mutable access to the neurons in this layer.
    * @return A reference to the vector of Neuron objects.
    * @warning Modifying neurons directly can affect network state. Use with caution.
    */
    [[nodiscard]] std::vector<Neuron>& get_neurons_mut() noexcept;

    /**
     * @brief Gets the activation function type used by neurons in this layer.
     * @return The ActivationType enum value for this layer.
     */
    [[nodiscard]] ActivationType get_activation_type() const noexcept;

    /**
     * @brief Gets the number of neurons in this layer.
     * @return The number of neurons.
     */
    [[nodiscard]] std::size_t size() const noexcept;
};

#endif // LAYER_H
