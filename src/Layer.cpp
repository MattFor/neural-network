// Layer.cpp
// Implementation file for the Layer class.

#include "../include/Layer.h"
#include "../include/Neuron.h" // Ensure Neuron definition is available

#include <vector>
#include <numeric>   // For std::iota (used via ranges)
#include <ranges>    // For std::views::iota
#include <algorithm> // For std::transform, std::for_each, std::clamp
#include <execution> // For std::execution::unseq
#include <stdexcept> // For std::invalid_argument, std::out_of_range
#include <format>    // For error messages (requires C++20)

// Constructor implementation
Layer::Layer(const std::size_t num_neurons,
             const std::size_t num_inputs_per_neuron,
             const double      weight_init_min,
             const double      weight_init_max)
    : outputs(num_neurons, 0.0) // Pre-allocate outputs vector
{
    neurons.reserve(num_neurons); // Reserve space for efficiency
    for (std::size_t i = 0; i < num_neurons; ++i)
    {
        // Call the corrected Neuron constructor
        neurons.emplace_back(num_inputs_per_neuron, weight_init_min, weight_init_max);
    }
}

// Feed-forward implementation
// Note: No longer const because Neuron::activate is not const.
const std::vector<double>& Layer::feed_forward(std::span<const double> inputs)
{
    // Input validation (optional but recommended)
    if (!neurons.empty() && inputs.size() != neurons[0].get_weights().size())
    {
        throw std::invalid_argument(std::format(
            "Layer::feed_forward: Input size ({}) does not match expected number of inputs per neuron ({}).",
            inputs.size(), neurons[0].get_weights().size()));
    }

    // Convert span to vector ONCE for Neuron::activate compatibility/efficiency
    // (Neuron::activate expects std::vector<double>)
    const std::vector<double> inputs_vec(inputs.begin(), inputs.end());

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
    const Layer*            next_layer_ptr) // Pointer to the *next* layer (or nullptr if output)
{
    const std::size_t num_neurons = this->neurons.size();
    if (num_neurons == 0) { return; } // Nothing to do

    // Input validation (optional but recommended)
    const std::size_t num_inputs_per_neuron = this->neurons[0].get_weights().size();
    if (prev_layer_outputs.size() != num_inputs_per_neuron)
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

    // --- 1.5 (Optional) Apply Gradient Clipping ---
    // If gradient clipping is desired, uncomment and ensure Neuron has set_gradient.

    // constexpr double GRADIENT_CLIP_THRESHOLD = 1.0; // Example threshold
    // if constexpr (GRADIENT_CLIP_THRESHOLD > 0) // Compile-time check if clipping enabled
    // {
    //      std::for_each(
    //         std::execution::unseq,
    //         this->neurons.begin(),
    //         this->neurons.end(),
    //         [](Neuron& neuron) {
    //              // Assumes Neuron class has set_gradient method
    //              const double clipped_gradient = std::clamp(
    //                  neuron.get_gradient(),
    //                 -GRADIENT_CLIP_THRESHOLD,
    //                  GRADIENT_CLIP_THRESHOLD
    //              );
    //              neuron.set_gradient(clipped_gradient);
    //         }
    //      );
    // }
    // --- End Gradient Clipping ---

    // --- 2. Update Weights using computed (and possibly clipped) gradients ---
    // Convert span to vector once for neuron.update_weights
    const std::vector<double> prev_outputs_vec(prev_layer_outputs.begin(), prev_layer_outputs.end());

    std::for_each(
        std::execution::unseq, // Potential parallelism
        this->neurons.begin(),
        this->neurons.end(),
        // Capture necessary variables by value/const ref for lambda.
        // Neuron must be non-const reference to allow update_weights to modify it.
        [learning_rate, &prev_outputs_vec](Neuron& neuron)
        {
            neuron.update_weights(prev_outputs_vec, learning_rate);
        }); // End weight update loop
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
