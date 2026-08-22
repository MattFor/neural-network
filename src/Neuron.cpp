//
// Created by MattFor on 12.04.2025.
//

#include <random>
#include <numeric>
#include <algorithm>
#include <execution>

#include "Neuron.h"

namespace
{
    /**
     * @brief The shared engine used to initialise weights and biases.
     */
    std::mt19937& weight_init_engine()
    {
        thread_local std::mt19937 engine{
            std::random_device{}()
        };

        return engine;
    }
}

// Constructor
Neuron::Neuron(const std::size_t num_inputs, const ActivationType activation, const double weight_init_min, const double weight_init_max)
    : bias(0.0), output(0.0), gradient(0.0), last_net_input(0.0), weights(num_inputs), activation_type(activation)
{
    std::uniform_real_distribution dis(weight_init_min, weight_init_max);
    std::mt19937&                  gen = weight_init_engine();

    // Initialize weights randomly
    for (double& weight : this->weights)
    {
        weight = dis(gen);
    }

    // Initialize bias randomly
    this->bias = dis(gen);
}

// Calculate neuron's output
double Neuron::activate(const std::span<const double> inputs)
{
    // --- Input Validation ---
    if (inputs.size() != this->weights.size())
    {
        throw std::invalid_argument(std::format("Neuron::activate: Input size ({}) does not match neuron's weight count ({}).", inputs.size(), this->weights.size()));
    }

    // --- Calculate Net Input (Weighted Sum + Bias) ---
    // Using std::transform_reduce for potential efficiency gains (vectorization/parallelism).
    // Calculates: sum(inputs[i] * weights[i]) for all i.
    const double weighted_sum = std::transform_reduce(
        // std::execution::unseq allows vectorization/unsequenced execution on a single thread.
        // Deliberately NOT par_unseq: this is the innermost loop of the network, so handing
        // each dot product to a thread pool costs far more than it saves.
        std::execution::unseq, inputs.begin(), inputs.end(), this->weights.begin(), 0.0 // Initial value for the sum
    );

    // Store the net input (weighted sum + bias). This is needed for derivatives
    // like ReLU and LeakyReLU that depend on the input value.
    this->last_net_input = weighted_sum + this->bias;

    // --- Apply Activation Function ---
    // Call the helper function which uses a switch based on this->activation_type
    // to apply the correct activation (Sigmoid, ReLU, LeakyReLU, Tanh).
    this->output = apply_activation(this->last_net_input);

    // Return the calculated activation value.
    return this->output;
}

// Compute gradient for an output layer neuron
void Neuron::compute_output_gradient(const double target)
{
    const double error = target - this->output;

    // The network reports binary cross-entropy loss (NeuralNetwork::compute_loss). With a
    // sigmoid output, dL/dz for that loss is exactly (output - target): the sigma'(z) factor
    // cancels against the 1/(p(1-p)) in dL/dp. Multiplying by apply_derivative() here would
    // therefore optimise squared error instead of the loss being reported, and would also
    // stall learning whenever a neuron saturates (sigma'(z) -> 0).
    if (this->activation_type == ActivationType::SIGMOID)
    {
        this->gradient = error;
        return;
    }

    // ReLU, LeakyReLU and Tanh outputs fall outside (0, 1), so cross-entropy does not apply
    // to them. They keep the squared-error delta rule: delta = (target - output) * f'(net).
    this->gradient = error * apply_derivative();
}


// Update weights and bias
void Neuron::update_weights(const std::span<const double> prev_layer_outputs, const double learning_rate)
{
    if (prev_layer_outputs.size() != this->weights.size())
    {
        throw std::invalid_argument(std::format("Neuron::update_weights: Previous layer output size ({}) does not match neuron's weight count ({}).", prev_layer_outputs.size(), this->weights.size()));
    }

    // Update weights w_ij = w_ij + learning_rate * gradient_j * output_i
    // Here, 'gradient_j' is this->gradient, 'output_i' is prev_layer_outputs[i]
    std::transform(std::execution::unseq,                                                                         // Allow unsequenced execution
                   this->weights.begin(), this->weights.end(), prev_layer_outputs.begin(), this->weights.begin(), // Write back to weights
                   [learning_rate, grad = this->gradient](const double w, const double prev_output)
                   {
                       // Capture gradient by value
                       return w + learning_rate * grad * prev_output;
                   });

    // Update bias: bias = bias + learning_rate * gradient
    // The 'input' for the bias is implicitly 1.0
    this->bias += learning_rate * this->gradient; // * 1.0
}

// --- Activation Function Implementations ---

double Neuron::activation_relu(const double x) noexcept
{
    return std::max(0.0, x);
}

double Neuron::activation_tanh(const double x) noexcept
{
    const double result = std::tanh(x);
    return std::clamp(result, -MAX_TANH_OUTPUT, MAX_TANH_OUTPUT);
}

double Neuron::activation_sigmoid(const double x) noexcept
{
    const double clamped_x = std::clamp(x, -MAX_EXP_INPUT, MAX_EXP_INPUT);
    return 1.0 / ( 1.0 + std::exp(-clamped_x) );
}

// Added LeakyReLU Activation
double Neuron::activation_leaky_relu(const double x) noexcept
{
    return x > 0.0 ? x : LRELU_ALPHA * x;
    // Equivalent: return std::max(LRELU_ALPHA * x, x); but branch might be faster
}


// --- Activation Derivative Implementations ---

double Neuron::derivative_relu(const double net_input) noexcept
{
    return net_input > 0.0 ? 1.0 : 0.0;
}

double Neuron::derivative_tanh(const double tanh_output) noexcept
{
    return 1.0 - tanh_output * tanh_output;
}

// Added LeakyReLU Derivative
double Neuron::derivative_leaky_relu(const double net_input) noexcept
{
    return net_input > 0.0 ? 1.0 : LRELU_ALPHA;
}

double Neuron::derivative_sigmoid(const double sigmoid_output) noexcept
{
    return sigmoid_output * ( 1.0 - sigmoid_output );
}

double Neuron::apply_activation(const double net_input) const noexcept
{
    switch (this->activation_type)
    {
        case ActivationType::SIGMOID:
        {
            return activation_sigmoid(net_input);
        }
        case ActivationType::RELU:
        {
            return activation_relu(net_input);
        }
        case ActivationType::LEAKY_RELU:
        {
            return activation_leaky_relu(net_input);
        }
        case ActivationType::TANH:
        {
            return activation_tanh(net_input);
        }
        default:
        {
            return 0.0; // Impossible
        }
    }
}

double Neuron::apply_derivative() const noexcept
{
    switch (this->activation_type)
    {
        case ActivationType::SIGMOID:
        {
            return derivative_sigmoid(this->output);
        }
        case ActivationType::RELU:
        {
            return derivative_relu(this->last_net_input);
        }
        case ActivationType::LEAKY_RELU:
        {
            return derivative_leaky_relu(this->last_net_input);
        }
        case ActivationType::TANH:
        {
            return derivative_tanh(this->output);
        }
        default:
        {
            return 0.0; // Impossible
        }
    }
}


// --- Getters ---

/** @brief Gets the last calculated output value. */
double Neuron::get_output() const noexcept
{
    return this->output;
}

/** @brief Gets the calculated gradient (delta). */
double Neuron::get_gradient() const noexcept
{
    return this->gradient;
}

/** @brief Gets the bias value. */
double Neuron::get_bias() const noexcept
{
    return this->bias;
}

/** @brief Gets the vector of input weights. */
const std::vector<double>& Neuron::get_weights() const noexcept
{
    return this->weights;
}

/** @brief Gets the vector of input weights. Non-const for setting and loading weights. */
std::vector<double>& Neuron::get_weights_mut() noexcept
{
    return this->weights;
}

// --- Setters ---

/** @brief For clipping the gradient. */
void Neuron::set_gradient(const double new_gradient) noexcept
{
    this->gradient = new_gradient;
}

/** @brief For restoring the bias when loading weights from a binary file. */
void Neuron::set_bias(const double new_bias) noexcept
{
    this->bias = new_bias;
}
