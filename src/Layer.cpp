//
// Created by MattFor on 12.04.2025.
//

#include "Layer.h"
#include "Neuron.h"

#include <vector>
#include <format>
#include <algorithm>
#include <stdexcept>

// Constructor implementation
Layer::Layer(const std::size_t num_neurons, std::size_t num_inputs_per_neuron, ActivationType activation, double weight_init_min, double weight_init_max)
    : outputs(num_neurons), layer_activation_type(activation)
{
    this->neurons.reserve(num_neurons);

    for (std::size_t i = 0; i < num_neurons; i++)
    {
        // Pass activation type and weight init params to Neuron constructor
        this->neurons.emplace_back(num_inputs_per_neuron, activation, weight_init_min, weight_init_max);
    }
}

// Feed-forward implementation
const std::vector<double>& Layer::feed_forward(const std::span<const double> inputs)
{
    // Input validation
    if (!this->neurons.empty() && inputs.size() != neurons[0].get_weights().size())
    {
        throw std::invalid_argument(std::format("Layer::feed_forward: Input size ({}) does not match expected number of inputs per neuron ({}).", inputs.size(), this->neurons[0].get_weights().size()));
    }

    for (std::size_t i = 0; i < this->neurons.size(); i++)
    {
        this->outputs[i] = this->neurons[i].activate(inputs);
    }

    return this->outputs; // Return reference to the updated internal cache
}

// --- Backpropagation, phase 1: gradients ---
void Layer::compute_gradients(const std::span<const double> error_signals,  // Targets (output layer) or ignored (hidden)
                              const Layer*                  next_layer_ptr, // Pointer to the *next* layer (or nullptr if output)
                              const double                  gradient_clip_threshold)
{
    const std::size_t num_neurons = this->neurons.size();
    if (num_neurons == 0)
    {
        return;
    }

    if (next_layer_ptr == nullptr && error_signals.size() != num_neurons)
    {
        throw std::out_of_range(std::format("Layer::compute_gradients: Output layer error signals size ({}) does not match neuron count ({}).", error_signals.size(), num_neurons));
    }

    for (std::size_t neuron_index = 0; neuron_index < num_neurons; neuron_index++) // neuron_index is 'j'
    {
        Neuron& current_neuron = this->neurons[neuron_index];

        if (next_layer_ptr == nullptr) // This is the Output Layer
        {
            // Gradient calculation: delta_j = dLoss/dnet_j (see Neuron::compute_output_gradient)
            current_neuron.compute_output_gradient(error_signals[neuron_index]);
        }
        else // This is a Hidden Layer
        {
            // Gradient calculation: delta_j = (Sum_k delta_k * w_jk) * f'(net_j)
            // Pass index 'j' and the next layer 'k'
            current_neuron.compute_hidden_gradient<Layer>(neuron_index, *next_layer_ptr);
        }
    }

    // --- Apply Gradient Clipping ---
    if (gradient_clip_threshold > 0.0)
    {
        for (Neuron& neuron : this->neurons)
        {
            neuron.set_gradient(std::clamp(neuron.get_gradient(), -gradient_clip_threshold, // Lower bound
                                           gradient_clip_threshold                          // Upper bound
            ));
        }
    }
}


// --- Backpropagation, phase 2: weight updates ---
void Layer::apply_weight_updates(const std::span<const double> prev_layer_outputs, // Outputs from previous layer
                                 const double                  learning_rate)
{
    if (this->neurons.empty())
    {
        return;
    }

    if (const std::size_t num_inputs_per_neuron = this->neurons[0].get_weights().size(); prev_layer_outputs.size() != num_inputs_per_neuron)
    {
        throw std::invalid_argument(std::format("Layer::apply_weight_updates: Previous layer output size ({}) does not match expected number of inputs per neuron ({}).", prev_layer_outputs.size(), num_inputs_per_neuron));
    }

    for (Neuron& neuron : this->neurons)
    {
        neuron.update_weights(prev_layer_outputs, learning_rate);
    }
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

std::vector<Neuron>& Layer::get_neurons_mut() noexcept
{
    return this->neurons;
}

/**
 * @brief Gets the activation function type used by neurons in this layer.
 */
ActivationType Layer::get_activation_type() const noexcept
{
    return this->layer_activation_type;
}

std::size_t Layer::size() const noexcept
{
    return this->neurons.size();
}
