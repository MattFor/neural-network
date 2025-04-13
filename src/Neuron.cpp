// Neuron.cpp
// Implementation file for the Neuron class.

#include "../include/Neuron.h" // Use quotes for local includes

// Constructor
Neuron::Neuron(const std::size_t num_inputs, const double weight_init_min, const double weight_init_max)
    : bias(0.0), // Initialize bias
      weights(num_inputs), // Allocate space for weights
      output(0.0),
      gradient(0.0)
{
    // Random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(weight_init_min, weight_init_max);

    // Initialize weights randomly
    for (double& weight : this->weights)
    {
        weight = dis(gen);
    }

    // Initialize bias randomly
    this->bias = dis(gen);
}

// Sigmoid activation function
double Neuron::sigmoid(const double x) noexcept
{
    // Clamp input to prevent overflow/underflow in std::exp
    const double clamped_x = std::clamp(x, -MAX_SIGMOID_INPUT, MAX_SIGMOID_INPUT);
    return 1.0 / (1.0 + std::exp(-clamped_x));
}

// Derivative of the sigmoid function
double Neuron::sigmoid_derivative(const double sigmoid_output) noexcept
{
    // derivative = sigmoid(x) * (1 - sigmoid(x))
    return sigmoid_output * (1.0 - sigmoid_output);
}

// Calculate neuron's output
double Neuron::activate(const std::vector<double>& inputs)
{
    if (inputs.size() != this->weights.size()) {
        throw std::invalid_argument(std::format(
            "Neuron::activate: Input size ({}) does not match neuron's weight count ({}).",
            inputs.size(), this->weights.size()));
    }

    // Calculate weighted sum: sum(inputs[i] * weights[i])
    // Using std::transform_reduce for potential parallelization/vectorization
    const double weighted_sum = std::transform_reduce(
        std::execution::unseq, // Allow unsequenced execution (potential vectorization)
        inputs.begin(),
        inputs.end(),
        this->weights.begin(),
        0.0 // Initial sum
    );

    // Add bias and apply activation function
    this->output = sigmoid(weighted_sum + this->bias);
    return this->output;
}

// Compute gradient for an output layer neuron
void Neuron::compute_output_gradient(const double target)
{
    // Gradient (delta) for output neuron = (target - output) * f'(net)
    // where f'(net) is sigmoid_derivative(output)
    const double error = target - this->output;
    this->gradient = error * sigmoid_derivative(this->output);
}


// Update weights and bias
void Neuron::update_weights(const std::vector<double>& prev_layer_outputs, const double learning_rate)
{
     if (prev_layer_outputs.size() != this->weights.size()) {
        throw std::invalid_argument(std::format(
            "Neuron::update_weights: Previous layer output size ({}) does not match neuron's weight count ({}).",
            prev_layer_outputs.size(), this->weights.size()));
    }

    // Update weights w_ij = w_ij + learning_rate * gradient_j * output_i
    // Here, 'gradient_j' is this->gradient, 'output_i' is prev_layer_outputs[i]
    std::transform(
        std::execution::unseq, // Allow unsequenced execution
        this->weights.begin(),
        this->weights.end(),
        prev_layer_outputs.begin(),
        this->weights.begin(), // Write back to weights
        [learning_rate, grad = this->gradient](double w, double prev_output) { // Capture gradient by value
            return w + learning_rate * grad * prev_output;
        }
    );

    // Update bias: bias = bias + learning_rate * gradient
    // The 'input' for the bias is implicitly 1.0
    this->bias += learning_rate * this->gradient; // * 1.0
}

// Note: The template function Neuron::compute_hidden_gradient is implemented in the header file (Neuron.h)
// because it's a template.