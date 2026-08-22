//
// Created by MattFor on 12.04.2025.
//

#ifndef NEURON_H
#define NEURON_H

#include <span>
#include <format>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <ranges>
#include <cstddef>
#include <concepts>
#include <stdexcept>

/**
 * @brief Enumeration of supported activation functions (applied per-neuron).
 * @note Softmax is handled at the Layer level due to interneuron dependency.
 */
enum class ActivationType
{
    SIGMOID,    ///< Sigmoid activation function (output range (0, 1))
    RELU,       ///< Rectified Linear Unit activation function (output range [0, inf))
    LEAKY_RELU, ///< Leaky Rectified Linear Unit activation function
    TANH        ///< Hyperbolic Tangent activation function (output range (-1, 1))
};

inline ActivationType int_to_activation(const int choice)
{
    switch (choice)
    {
        case 1:
        {
            return ActivationType::RELU;
        }
        case 2:
        {
            return ActivationType::LEAKY_RELU;
        }
        case 3:
        {
            return ActivationType::TANH;
        }
        case 4:
        {
            return ActivationType::SIGMOID;
        }
        default:
        {
            throw std::invalid_argument("Invalid activation function choice number: " + std::to_string(choice));
        }
    }
}

inline std::string activation_to_string(const ActivationType type)
{
    switch (type)
    {
        case ActivationType::SIGMOID:
        {
            return "Sigmoid";
        }

        case ActivationType::RELU:
        {
            return "ReLU";
        }

        case ActivationType::LEAKY_RELU:
        {
            return "LeakyReLU";
        }

        case ActivationType::TANH:
        {
            return "Tanh";
        }

        default:
        {
            return "Unknown";
        }
    }
}

// Clamping constants to prevent overflow/underflow issues.

/// ~log(DBL_MAX); bounds the exponent passed to std::exp in the sigmoid.
inline constexpr double MAX_EXP_INPUT = 709.0;

/// The largest double strictly below 1.0, used to keep tanh away from exactly +/-1.0 (where its derivative 1 - y^2 would collapse to zero).
inline constexpr double MAX_TANH_OUTPUT = 1.0 - std::numeric_limits<double>::epsilon() / 2.0;

/// Standard alpha for Leaky ReLU.
inline constexpr double LRELU_ALPHA = 0.01;

/**
 * @brief Represents a single artificial neuron for use in a neural network.
 *
 * Holds one weight per input plus a bias, applies a configurable activation function, and caches the values backpropagation needs (net input, output, delta).
 */
class Neuron
{
    double bias;           ///< Bias value for this neuron.
    double output;         ///< Stores the last calculated output (activation) value.
    double gradient;       ///< Stores the gradient (delta) calculated during backpropagation.
    double last_net_input; ///< Stores the last net input (weighted_sum + bias) needed for some derivatives.

    std::vector<double> weights; ///< Weights connecting inputs from the previous layer.

    ActivationType activation_type;

    // --- Activation Functions (Static Private Helpers) ---
    [[nodiscard]] static double activation_relu(double x) noexcept;
    [[nodiscard]] static double activation_tanh(double x) noexcept;
    [[nodiscard]] static double activation_sigmoid(double x) noexcept;
    [[nodiscard]] static double activation_leaky_relu(double x) noexcept;

    // --- Activation Derivatives (Static Private Helpers) ---
    [[nodiscard]] static double derivative_relu(double net_input) noexcept;
    [[nodiscard]] static double derivative_tanh(double tanh_output) noexcept;
    [[nodiscard]] static double derivative_leaky_relu(double net_input) noexcept;
    [[nodiscard]] static double derivative_sigmoid(double sigmoid_output) noexcept;

    /** @brief Applies the neuron's configured activation function to the net input. */
    [[nodiscard]] double apply_activation(double net_input) const noexcept;

    /** @brief Applies the derivative of the neuron's configured activation function. */
    [[nodiscard]] double apply_derivative() const noexcept;

public:
    /**
     * @brief Constructs a Neuron.
     *
     * @param num_inputs The number of inputs connecting to this neuron (size of the previous layer).
     * @param activation Which activation function to use, default: SIGMOID
     * @param weight_init_min Minimum value for random weight/bias initialization.
     * @param weight_init_max Maximum value for random weight/bias initialization.
     *
     * @note Weights and bias are initialized randomly within the specified range.
     */
    explicit Neuron(std::size_t num_inputs, ActivationType activation = ActivationType::SIGMOID, double weight_init_min = -0.5, double weight_init_max = 0.5);

    /**
     * @brief Calculates the neuron's output based on inputs from the previous layer.
     *
     * Computes the weighted sum of inputs plus the bias (net input), stores
     * this net input, applies the neuron's configured activation function,
     * stores the result, and returns it.
     *
     * @param inputs A view of the output values from the neurons in the previous layer.
     *
     * @return The calculated activation value (also stored internally).
     *
     * @throws std::invalid_argument if input size doesn't match the number of weights.
     */
    double activate(std::span<const double> inputs);

    /**
     * @brief Computes the gradient for this neuron if it's in the output layer.
     *
     * The stored gradient is the negated derivative of the loss with respect to this
     * neuron's net input, so that update_weights() can simply add
     * `learning_rate * gradient * input`.
     *
     * For a SIGMOID output paired with the binary cross-entropy loss reported by
     * NeuralNetwork::compute_loss, the sigmoid derivative cancels out of dL/dz exactly,
     * leaving `delta = target - output`.
     *
     * Any other output activation has a range outside (0, 1) and is therefore not a valid
     * input to cross-entropy; those fall back to the squared-error delta rule
     * `delta = (target - output) * f'(net_input)`.
     *
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
     *
     * @note The gradient is stored internally. Requires C++20 for range constraints.
     *
     * @throws std::out_of_range if `index_in_this_layer` is invalid for a neuron in the next layer.
     */
    template <typename LayerType> requires requires(const LayerType l, const std::ranges::range_value_t<decltype(l.get_neurons())> n)
    {
        { l.get_neurons() } -> std::ranges::range; // Must have get_neurons() returning a range
        { n.get_weights() } -> std::ranges::range; // Neurons in the range must have get_weights() returning a range
        { n.get_gradient() } -> std::convertible_to<double>;
        // Neurons must have get_gradient() returning double-convertible
    }
    void compute_hidden_gradient(std::size_t index_in_this_layer, const LayerType& next_layer);


    /**
     * @brief Updates the neuron's weights and bias based on the calculated gradient.
     *
     * @param prev_layer_outputs A view of the output values from the previous layer (inputs to this neuron).
     * @param learning_rate The learning rate hyperparameter.
     *
     * @throws std::invalid_argument if prev_layer_outputs size doesn't match the number of weights.
     */
    void update_weights(std::span<const double> prev_layer_outputs, double learning_rate);

    // --- Getters ---

    /** @brief Gets the last calculated output value. */
    [[nodiscard]] double get_output() const noexcept;

    /** @brief Gets the calculated gradient (delta). */
    [[nodiscard]] double get_gradient() const noexcept;

    /** @brief Gets the bias value. */
    [[nodiscard]] double get_bias() const noexcept;

    /** @brief Gets the vector of input weights. */
    [[nodiscard]] const std::vector<double>& get_weights() const noexcept;

    /** @brief Gets the vector of input weights. Non-const for setting and loading weights. */
    [[nodiscard]] std::vector<double>& get_weights_mut() noexcept;

    // --- Setters ---

    /** @brief For clipping the gradient. */
    void set_gradient(double new_gradient) noexcept;

    /** @brief For restoring the bias when loading weights from a binary file. */
    void set_bias(double new_bias) noexcept;
};

/**
 * @brief Computes the gradient for this neuron if it's in a hidden layer.
 *
 * Calculates the gradient (delta) for a hidden neuron 'j' using the formula:
 * delta_j = (Sum_k w_jk * delta_k) * f'(net_j)
 * where 'k' represents neurons in the next layer, w_jk is the weight connecting
 * neuron 'j' to neuron 'k', delta_k is the gradient of neuron 'k', and f'(net_j)
 * is the derivative of this neuron's activation function evaluated at its net input.
 * The specific derivative function used depends on the neuron's configured activation type.
 *
 * (See Neuron.h for detailed template parameter and exception documentation)
 *
 * @tparam LayerType The type of the next layer in the network. Must satisfy constraints defined in Neuron.h.
 * @param index_in_this_layer The index 'j' of this neuron within its own layer.
 * @param next_layer A constant reference to the layer immediately following this one.
 */
template <typename LayerType> requires requires(const LayerType l, const std::ranges::range_value_t<decltype(l.get_neurons())> n)
{
    // Constraints ensure the LayerType and its neurons provide necessary methods
    { l.get_neurons() } -> std::ranges::range;           // Must be able to get neurons as a range
    { n.get_weights() } -> std::ranges::range;           // Neurons must provide weights as a range
    { n.get_gradient() } -> std::convertible_to<double>; // Neurons must provide a gradient value
}

void Neuron::compute_hidden_gradient(std::size_t index_in_this_layer, const LayerType& next_layer)
{
    // --- Calculate Sum of Weighted Gradients from Next Layer ---
    double      sum_weighted_gradients = 0.0;
    const auto& next_neurons           = next_layer.get_neurons(); // Get neurons from the layer ahead
    std::size_t k                      = 0;                        // Index for debugging/error messages

    // Iterate through each neuron 'k' in the next layer
    for (const auto& next_neuron : next_neurons)
    {
        // Get the weights vector of the k-th neuron in the next layer
        const auto& weights_of_next_neuron = next_neuron.get_weights();
        const auto  num_weights_in_next    = std::ranges::size(weights_of_next_neuron);

        // --- Topology Validation ---
        // Ensure the current neuron's index ('j') is valid for the weight vector of the next neuron ('k')
        if (index_in_this_layer >= num_weights_in_next)
        {
            // This indicates a mismatch in expected network structure
            throw std::out_of_range(std::format("Neuron::compute_hidden_gradient: Index of this neuron ({}) is out of bounds " "for the weights of neuron {} in the next layer (size {}). " "Layer topology mismatch detected.", index_in_this_layer, k, num_weights_in_next));
        }

        // --- Access Weight w_jk ---
        // Get the specific weight connecting this neuron ('j') to the next neuron ('k').
        // Using iterators for safety, as get_weights() returns a range.
        auto weight_it = std::ranges::begin(weights_of_next_neuron);
        std::ranges::advance(weight_it, index_in_this_layer); // Move iterator to the j-th weight
        const double weight_jk = *weight_it;                  // Dereference to get the weight value

        // Get the gradient (delta_k) already computed for the k-th neuron in the next layer
        const double gradient_k = next_neuron.get_gradient();

        // Add the contribution to the sum: w_jk * delta_k
        sum_weighted_gradients += weight_jk * gradient_k;
        k++; // Increment index for next neuron in the next layer
    }

    // --- Calculate Final Gradient for This Hidden Neuron ---
    // Gradient (delta) for hidden neuron 'j' = sum(w_jk * delta_k) * f'(net_j)
    // where f'(net_j) is the derivative of this neuron's activation function.
    // Then use the apply_derivative() helper which selects the correct derivative
    // based on this->activation_type (using this->output or this->last_net_input as needed).
    this->gradient = sum_weighted_gradients * apply_derivative();
}

#endif // NEURON_H
