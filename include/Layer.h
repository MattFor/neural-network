//
// Created by MattFor on 12.04.2025.
//

#ifndef LAYER_H
#define LAYER_H

#include <span>
#include <vector>
#include <cstddef>

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
     *
     * @param num_neurons The number of neurons in this layer.
     * @param num_inputs_per_neuron The number of inputs connecting to EACH neuron
     * (i.e., the number of neurons in the previous layer).
     * @param activation Which activation function should be used for neurons.
     * @param weight_init_min Minimum value for random weight/bias initialization (passed to neurons).
     * @param weight_init_max Maximum value for random weight/bias initialization (passed to neurons).
     */
    Layer(std::size_t num_neurons, std::size_t num_inputs_per_neuron, ActivationType activation, double weight_init_min = -0.5, double weight_init_max = 0.5);

    /**
     * @brief Performs a feed-forward pass for this layer.
     *
     * Calculates the output of each neuron based on the inputs and updates
     * the internal output cache.
     *
     * @param inputs A view of the output vector from the previous layer (or the network's input).
     *
     * @return A reference to the internally cached vector of output activations for this layer.
     *
     * @throws std::invalid_argument if the size of inputs does not match num_inputs_per_neuron.
     *
     * @note This method is not const because it calls the non-const Neuron::activate.
     */
    [[nodiscard]] const std::vector<double>& feed_forward(std::span<const double> inputs);

    /**
     * @brief Computes the gradient (delta) of every neuron in this layer.
     *
     * This is the first half of backpropagation. A hidden layer's gradients are derived from the weights of
     * the layer in front of it, so *every* layer must compute its gradient from the
     * weights used during the forward pass, before *any* layer updates them.
     *
     * @param error_signals For the output layer, the target values corresponding to each
     * neuron in this layer. Ignored for hidden layers, whose gradients come from `next_layer_ptr` instead.
     * @param next_layer_ptr A pointer to the next layer (in the forward direction).
     * If `nullptr`, this layer is treated as the output layer.
     * @param gradient_clip_threshold Clamps each delta to [-threshold, +threshold]. Pass 0.0 (or less) to disable clipping.
     *
     * @throws std::out_of_range If treated as the output layer and `error_signals.size()` doesn't match the neuron count.
     */
    void compute_gradients(std::span<const double> error_signals, const Layer* next_layer_ptr, double gradient_clip_threshold = 0.0);

    /**
     * @brief Applies the already-computed gradients to this layer's weights and biases.
     *
     * This is the second half of backpropagation; call compute_gradients() on every layer
     * of the network first.
     *
     * @param prev_layer_outputs A view of the activation outputs from the PREVIOUS layer
     * (which served as inputs during the feed_forward pass).
     * Its size must match num_inputs_per_neuron.
     * @param learning_rate The learning rate for weight updates.
     *
     * @throws std::invalid_argument If `prev_layer_outputs.size()` doesn't match `num_inputs_per_neuron`.
     */
    void apply_weight_updates(std::span<const double> prev_layer_outputs, double learning_rate);

    // --- Getters ---

    /**
     * @brief Gets the cached output activations from the last feed-forward pass.
     *
     * @return A constant reference to the vector of output values.
     */
    [[nodiscard]] const std::vector<double>& get_outputs() const noexcept;

    /**
     * @brief Provides access to the neurons in this layer.
     *
     * @return A constant reference to the vector of Neuron objects.
     */
    [[nodiscard]] const std::vector<Neuron>& get_neurons() const noexcept;

    /**
    * @brief Provides mutable access to the neurons in this layer.
    *
    * @return A reference to the vector of Neuron objects.
    *
    * @warning Modifying neurons directly can affect network state. Use with caution.
    */
    [[nodiscard]] std::vector<Neuron>& get_neurons_mut() noexcept;

    /**
     * @brief Gets the activation function type used by neurons in this layer.
     *
     * @return The ActivationType enum value for this layer.
     */
    [[nodiscard]] ActivationType get_activation_type() const noexcept;

    /**
     * @brief Gets the number of neurons in this layer.
     *
     * @return The number of neurons.
     */
    [[nodiscard]] std::size_t size() const noexcept;
};

#endif // LAYER_H
