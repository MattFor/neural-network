//
// Created by MattFor on 12.04.2025.
//

// Layer.cpp
// Implementation file for the Layer class.

#include "../include/Layer.h"
#include "../include/Neuron.h" // Ensure Neuron definition is available

#include <vector>
#include <ranges>    // For std::views::iota
#include <format>    // For error messages (requires C++20)
#include <numeric>   // For std::iota (used via ranges)
#include <algorithm> // For std::transform, std::for_each, std::clamp
#include <execution> // For std::execution::unseq
#include <stdexcept> // For std::invalid_argument, std::out_of_range

// Constructor implementation
Layer::Layer(
    const std::size_t num_neurons,
    std::size_t       num_inputs_per_neuron,
    ActivationType    activation,
    double            weight_init_min,
    double            weight_init_max)
    : outputs(num_neurons),
      layer_activation_type(activation)
{
    this->neurons.reserve(num_neurons);

    for (std::size_t i = 0; i < num_neurons; i++)
    {
        // Pass activation type and weight init params to Neuron constructor
        this->neurons.emplace_back(num_inputs_per_neuron, activation, weight_init_min, weight_init_max);
    }
}

/**
 * @brief Gets the activation function type used by neurons in this layer.
 */
ActivationType Layer::get_activation_type() const noexcept
{
    return this->layer_activation_type;
}

// Feed-forward implementation
// Note: No longer const because Neuron::activate is not const.
const std::vector<double>& Layer::feed_forward(std::span<const double> inputs)
{
    // Input validation (optional but recommended)
    if (!this->neurons.empty() && inputs.size() != neurons[0].get_weights().size())
    {
        throw std::invalid_argument(std::format(
            "Layer::feed_forward: Input size ({}) does not match expected number of inputs per neuron ({}).",
            inputs.size(), this->neurons[0].get_weights().size()));
    }

    // Convert span to vector ONCE for Neuron::activate compatibility/efficiency
    // (Neuron::activate expects std::vector<double>)
    const std::vector inputs_vec(inputs.begin(), inputs.end());

    // Compute outputs for each neuron in parallel.
    // Output iterator writes directly into the pre-allocated this->outputs vector.
    std::transform(
        std::execution::unseq, // Enable vectorization/parallelism potential
        this->neurons.begin(),
        this->neurons.end(),
        this->outputs.begin(),
        // Capture input vector by const reference.
        // Lambda takes Neuron& (non-const) because activate() is non-const.
        [&inputs_vec](Neuron& neuron) // Note: Neuron& is non-const
        {
            return neuron.activate(inputs_vec); // Calls non-const Neuron::activate
        }
    );

    return this->outputs; // Return reference to the updated internal cache
}


// Backpropagation implementation
void Layer::backpropagate(
    std::span<const double> error_signals,      // Targets (output layer) or ignored (hidden)
    std::span<const double> prev_layer_outputs, // Outputs from previous layer
    double                  learning_rate,
    const Layer*            next_layer_ptr,               // Pointer to the *next* layer (or nullptr if output)
    double                  gradient_clip_threshold = 0.0 // Default is 0.0 (no clipping)
)
{
    const std::size_t num_neurons = this->neurons.size();
    if (num_neurons == 0)
    {
        return;
    }

    // Input validation (optional but recommended)
    if (const std::size_t num_inputs_per_neuron = this->neurons[0].get_weights().size();
        prev_layer_outputs.size() != num_inputs_per_neuron)
    {
        throw std::invalid_argument(std::format(
            "Layer::backpropagate: Previous layer output size ({}) does not match expected number of inputs per neuron ({}).",
            prev_layer_outputs.size(), num_inputs_per_neuron));
    }

    if (next_layer_ptr == nullptr && error_signals.size() != num_neurons)
    {
        throw std::out_of_range(std::format(
            "Layer::backpropagate: Output layer error signals size ({}) does not match neuron count ({}).",
            error_signals.size(), num_neurons));
    }

    // --- 1. Compute Gradients (Deltas) for each neuron in this layer ---
    // Use std::views::iota for indices needed by compute_hidden_gradient
    auto index_view = std::views::iota(std::size_t{0}, num_neurons);

    std::for_each(
        std::execution::unseq, // Potential parallelism
        index_view.begin(),
        index_view.end(),
        [this, error_signals, next_layer_ptr](const std::size_t neuron_index) // neuron_index is 'j'
        {
            Neuron& current_neuron = this->neurons[neuron_index]; // Get non-const reference

            if (next_layer_ptr == nullptr) // This is the Output Layer
            {
                // Gradient calculation: delta_j = (target_j - output_j) * f'(net_j)
                current_neuron.compute_output_gradient(error_signals[neuron_index]);
            }
            else // This is a Hidden Layer
            {
                // Gradient calculation: delta_j = (Sum_k delta_k * w_jk) * f'(net_j)
                // Pass index 'j' and the next layer 'k'
                current_neuron.compute_hidden_gradient<Layer>(neuron_index, *next_layer_ptr);
            }
        }); // End gradient computation loop

    // --- 1.5 Apply Gradient Clipping (if enabled) ---
    // This clips the *delta* (gradient of the error w.r.t. the neuron's net input)
    // for each neuron individually.
    if (gradient_clip_threshold > 0.0)
    {
        std::for_each(
            std::execution::par_unseq, // Can be parallel
            this->neurons.begin(),
            this->neurons.end(),
            [gradient_clip_threshold](Neuron& neuron)
            {
                // Assumes Neuron class has get_gradient() and set_gradient() methods
                // Clamps the gradient value between [-threshold, +threshold]
                const double clipped_gradient = std::clamp(
                    neuron.get_gradient(),
                    -gradient_clip_threshold, // Lower bound
                    gradient_clip_threshold   // Upper bound
                );

                neuron.set_gradient(clipped_gradient); // Update the neuron's stored gradient
            }
        );
    }
    // --- End Gradient Clipping ---

    // --- 2. Update Weights using computed (and possibly clipped) gradients ---
    // Convert span to vector once for neuron.update_weights
    const std::vector prev_outputs_vec(prev_layer_outputs.begin(), prev_layer_outputs.end());

    std::for_each(
        std::execution::unseq, // Potential parallelism
        this->neurons.begin(),
        this->neurons.end(),
        // Capture necessary variables by value/const ref for lambda.
        // Neuron must be non-const reference to allow update_weights to modify it.
        [learning_rate, &prev_outputs_vec](Neuron& neuron)
        {
            neuron.update_weights(prev_outputs_vec, learning_rate);
        }
    ); // End weight update loop
}

// Getters implementation
const std::vector<double>& Layer::get_outputs() const noexcept
{
    return this->outputs;
}

const std::vector<Neuron>& Layer::get_neurons() const noexcept
{
    return this->neurons;
}

// Added getter for mutable access if needed
std::vector<Neuron>& Layer::get_neurons_mut() noexcept
{
    return this->neurons;
}

std::size_t Layer::size() const noexcept
{
    return this->neurons.size();
}
