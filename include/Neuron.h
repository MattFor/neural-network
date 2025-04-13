// Neuron.h
// Represents a single neuron in a neural network layer.

#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cstddef> // For std::size_t
#include <cmath>   // For std::exp, std::clamp
#include <random>  // For weight initialization
#include <stdexcept> // For exceptions
#include <numeric>   // For std::transform_reduce
#include <execution> // For std::execution::unseq
#include <algorithm> // For std::transform
#include <functional> // For std::plus
#include <ranges>     // For concepts/range checks (requires C++20)
#include <concepts>   // For std::convertible_to (requires C++20)
#include <format>     // For error messages (requires C++20)
#include <span>       // If needed for LayerType get_weights() interface

// Prevents overflow/underflow in std::exp during sigmoid calculation.
static constexpr double MAX_SIGMOID_INPUT = 34.538776394910684; // Approx log(DBL_MAX)

/**
 * @brief Represents a single artificial neuron for use in a neural network.
 *
 * This neuron uses the sigmoid activation function and is designed for
 * use in feedforward networks trained with backpropagation.
 */
class Neuron
{
private:
    double              bias;     ///< Bias value for this neuron.
    std::vector<double> weights;  ///< Weights connecting inputs from the previous layer.
    double              output;   ///< Stores the last calculated output (activation) value.
    double              gradient; ///< Stores the gradient (delta) calculated during backpropagation.

    /**
     * @brief Sigmoid activation function.
     * @param x The input value (weighted sum + bias).
     * @return The activated value between 0 and 1.
     * @note Clamps input to [-MAX_SIGMOID_INPUT, MAX_SIGMOID_INPUT] to prevent std::exp overflow.
     */
    [[nodiscard]] static double sigmoid(double x) noexcept;

    /**
     * @brief Derivative of the sigmoid function.
     * @param sigmoid_output The output of the sigmoid function (sigmoid(x)).
     * @return The value of the sigmoid derivative at x.
     * @note Takes sigmoid_output as input for efficiency, as output is often already computed.
     */
    [[nodiscard]] static double sigmoid_derivative(double sigmoid_output) noexcept;

public:
    /**
     * @brief Constructs a Neuron.
     * @param num_inputs The number of inputs connecting to this neuron (size of the previous layer).
     * @param weight_init_min Minimum value for random weight/bias initialization.
     * @param weight_init_max Maximum value for random weight/bias initialization.
     * @note Weights and bias are initialized randomly within the specified range.
     */
    explicit Neuron(std::size_t num_inputs, double weight_init_min = -0.5, double weight_init_max = 0.5);

    /**
     * @brief Calculates the neuron's output based on inputs from the previous layer.
     * @param inputs A vector of output values from the neurons in the previous layer.
     * @return The calculated activation value (also stored internally).
     * @throws std::invalid_argument if input size doesn't match the number of weights.
     */
    double activate(const std::vector<double>& inputs);

    /**
     * @brief Computes the gradient for this neuron if it's in the output layer.
     * @param target The target value for this neuron.
     * @note The gradient is stored internally.
     */
    void compute_output_gradient(double target);

    /**
     * @brief Computes the gradient for this neuron if it's in a hidden layer.
     *
     * @tparam LayerType The type of the next layer in the network. Must provide
     * a `get_neurons()` method returning a range of objects
     * that have `get_weights()` and `get_gradient()` methods.
     * @param index_in_this_layer The index 'j' of this neuron within its own layer.
     * This index is used to find the corresponding weight
     * in neurons of the *next* layer.
     * @param next_layer          A constant reference to the layer immediately following
     * this neuron's layer in the network ('k').
     * @note The gradient is stored internally. Requires C++20 for range constraints.
     * @throws std::out_of_range if `index_in_this_layer` is invalid for a neuron in the next layer.
     */
    template <typename LayerType>
    // C++20 Concept to constrain LayerType (optional but recommended)
        requires requires(const LayerType l, const std::ranges::range_value_t<decltype(l.get_neurons())> n)
        {
            { l.get_neurons() } -> std::ranges::range; // Must have get_neurons() returning a range
            { n.get_weights() } -> std::ranges::range; // Neurons in the range must have get_weights() returning a range
            { n.get_gradient() } -> std::convertible_to<double>;
            // Neurons must have get_gradient() returning double-convertible
        }
    void compute_hidden_gradient(std::size_t index_in_this_layer, const LayerType& next_layer);


    /**
     * @brief Updates the neuron's weights and bias based on the calculated gradient.
     * @param prev_layer_outputs The vector of output values from the previous layer (inputs to this neuron).
     * @param learning_rate The learning rate hyperparameter.
     * @throws std::invalid_argument if prev_layer_outputs size doesn't match the number of weights.
     */
    void update_weights(const std::vector<double>& prev_layer_outputs, double learning_rate);

    // --- Getters ---

    /** @brief Gets the last calculated output value. */
    [[nodiscard]] double get_output() const noexcept { return output; }

    /** @brief Gets the calculated gradient (delta). */
    [[nodiscard]] double get_gradient() const noexcept { return gradient; }

    /** @brief Gets the bias value. */
    [[nodiscard]] double get_bias() const noexcept { return bias; }

    /** @brief Gets the vector of input weights. */
    [[nodiscard]] const std::vector<double>& get_weights() const noexcept { return weights; }
};


// --- Template Implementation for compute_hidden_gradient ---
// Needs to be in the header file or included at the end of the header if definitions are separated.

template <typename LayerType>
// Add the requires clause again if not using separate include for template definitions
    requires requires(const LayerType l, const std::ranges::range_value_t<decltype(l.get_neurons())> n)
    {
        { l.get_neurons() } -> std::ranges::range;
        { n.get_weights() } -> std::ranges::range;
        { n.get_gradient() } -> std::convertible_to<double>;
    }
void Neuron::compute_hidden_gradient(std::size_t index_in_this_layer, const LayerType& next_layer)
{
    double      sum_weighted_gradients = 0.0;
    const auto& next_neurons           = next_layer.get_neurons(); // Get the range of neurons in the next layer

    // Iterate through neurons 'k' in the next layer
    std::size_t k = 0; // Keep track of neuron index in next layer for error messages
    for (const auto& next_neuron : next_neurons)
    {
        // Get weights w_jk (weight from neuron 'j' [this] to neuron 'k' [next_neuron])
        const auto& weights_of_next_neuron = next_neuron.get_weights(); // Requires NeuronType::get_weights()

        // Use std::ranges::size or .size() depending on what get_weights() returns
        const auto num_weights_in_next = std::ranges::size(weights_of_next_neuron);
        // const auto num_weights_in_next = weights_of_next_neuron.size(); // Alternative if it's a container

        if (index_in_this_layer >= num_weights_in_next)
        {
            throw std::out_of_range(std::format(
                "Neuron::compute_hidden_gradient: Index of this neuron ({}) is out of bounds "
                "for the weights of neuron {} in the next layer (size {}). "
                "Layer topology mismatch detected.",
                index_in_this_layer, k, num_weights_in_next));
        }

        // Access weight w_jk. This assumes weights are indexable or uses std::ranges:: C++20 access
        // Using iterators is safer if the exact type returned by get_weights() isn't known
        auto weight_it = std::ranges::begin(weights_of_next_neuron);
        std::ranges::advance(weight_it, index_in_this_layer);
        const double weight_jk = *weight_it;
        // const double weight_jk = weights_of_next_neuron[index_in_this_layer]; // Alternative if indexable

        const double gradient_k = next_neuron.get_gradient(); // Requires NeuronType::get_gradient()
        sum_weighted_gradients += weight_jk * gradient_k;
        ++k;
    }

    // Gradient (delta) for hidden neuron 'j' = sum(w_jk * delta_k) * f'(net_j)
    // where f'(net_j) is sigmoid_derivative(output)
    this->gradient = sum_weighted_gradients * sigmoid_derivative(this->output);
}


#endif // NEURON_H
