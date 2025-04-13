// NeuralNetwork.cpp
// Implementation file for the NeuralNetwork class.

// Include header for the class itself first (good practice)
#include "../include/NeuralNetwork.h" // Adjust path as needed

// Include necessary standard library headers
#include <span>
#include <vector>
#include <cmath>     // For std::log, std::max, std::min in compute_loss
#include <numeric>   // Potentially for std::accumulate, std::iota
#include <stdexcept> // For std::runtime_error, std::invalid_argument
#include <algorithm> // For std::min, std::max (if not using <cmath>)
#include <chrono>    // For timing in train
#include <iostream>  // For std::cout in display_topology
#include <iomanip>   // For std::setw, std::left in display_topology
#include <limits>    // For std::numeric_limits (used in compute_loss epsilon definition conceptually)
#include <string>    // For std::to_string, std::string
#include <cstdio>    // For std::printf, std::fflush in train
#include <typeinfo>  // For typeid in display_topology
#include <format>    // For std::format (requires C++20)

// Implementation for display_topology
void NeuralNetwork::display_topology() const
{
    std::cout << "\n--- Neural Network Topology ---\n\n";
    std::cout << "Input Size: " << this->input_size << "\n";
    std::cout << "-----------------------------\n";

    // Define column widths for formatting
    constexpr int     col_width_index   = 7;
    constexpr int     col_width_neurons = 9;
    constexpr int     col_width_inputs  = 15;
    constexpr int     col_width_type    = 12; // Adjusted for typical Layer name
    const std::string header_sep        = " | ";
    const std::string row_sep           = "| ";

    // Print table header
    std::cout << std::left
            << row_sep << std::setw(col_width_index) << "Layer #"
            << header_sep << std::setw(col_width_type) << "Type"
            << header_sep << std::setw(col_width_neurons) << "Neurons"
            << header_sep << std::setw(col_width_inputs) << "#Inputs/Neuron"
            << " |\n";

    // Print separator line based on calculated width
    const int total_width = col_width_index + col_width_type + col_width_neurons + col_width_inputs
                            + header_sep.length() * 3 + row_sep.length() + 2; // +2 for start/end |
    std::cout << std::string(total_width, '-') << "\n";


    // Iterate through layers
    for (std::size_t i = 0; i < this->layers.size(); ++i)
    {
        const auto&       layer                 = this->layers[i];
        const std::size_t num_neurons           = layer.size(); // Use layer.size()
        std::size_t       num_inputs_per_neuron = (i == 0) ? this->input_size : this->layers[i - 1].size();

        // Simplify type display - typeid().name() can be mangled
        std::string layer_type = (i == this->layers.size() - 1) ? "Output" : "Hidden";

        std::cout << std::left
                << row_sep << std::setw(col_width_index) << i
                << header_sep << std::setw(col_width_type) << layer_type
                << header_sep << std::setw(col_width_neurons) << num_neurons
                << header_sep << std::setw(col_width_inputs) << num_inputs_per_neuron
                << " |\n";

        // Optional: Display sample weights/bias from the first neuron
        if (!layer.get_neurons().empty())
        {
            const auto&  neuron  = layer.get_neurons().front();
            const auto&  weights = neuron.get_weights();
            const double bias    = neuron.get_bias();
            std::cout << row_sep << std::string(col_width_index, ' ') // Indent
                    << " Sample W: [";
            for (size_t w_idx = 0; w_idx < std::min(weights.size(), size_t{3}); ++w_idx)
            {
                std::cout << std::fixed << std::setprecision(3) << weights[w_idx] << (
                    w_idx < 2 && w_idx < weights.size() - 1 ? ", " : "");
            }
            if (weights.size() > 3)
                std::cout << ", ...";
            std::cout << "], Bias: " << std::fixed << std::setprecision(3) << bias << "\n";
        }
        std::cout << std::string(total_width, '-') << "\n";
    }
    std::cout << "Output Size: " << this->output_size << "\n";
    std::cout << "-----------------------------\n\n";
}


// Implementation for feed_forward (single sample)
// Note: Removed 'const'
std::vector<double> NeuralNetwork::feed_forward(std::span<const double> input)
{
    if (input.size() != this->input_size)
    {
        throw std::runtime_error(std::format(
            "NeuralNetwork::feed_forward: Input size mismatch. Expected {}, got {}.",
            this->input_size, input.size()));
    }

    // Use a temporary vector to hold activations, starting with the input
    // Copy input span to activation vector
    std::vector<double> current_activations(input.begin(), input.end());

    // Propagate through each layer
    // Must use non-const reference to layer because layer.feed_forward is non-const
    for (Layer& layer : this->layers)
    {
        // feed_forward updates the layer's internal outputs and returns a const ref
        // We need to copy the result for the next layer's input
        // Layer::feed_forward now takes a span, simplifying things
        const std::vector<double>& layer_output_ref = layer.feed_forward(current_activations);
        // Prepare for next iteration or final return
        current_activations = layer_output_ref; // Copy output for next layer's input or return
    }

    // The final activations after the last layer
    return current_activations;
}

// Implementation for feed_forward (batch)
// Note: Removed 'const'
std::vector<std::vector<double>> NeuralNetwork::feed_forward(const std::vector<std::vector<double>>& inputs)
{
    std::vector<std::vector<double>> results;
    results.reserve(inputs.size());

    // Process each sample individually
    for (const auto& sample : inputs)
    {
        // No need to check size here again if single feed_forward does it
        // std::span constructor performs no allocation
        results.push_back(feed_forward(std::span<const double>(sample)));
    }

    return results;
}

// Implementation for backpropagate (single sample)
void NeuralNetwork::backpropagate(
    std::span<const double> network_input,
    std::span<const double> target,
    double                  learning_rate)
{
    // Basic size checks
    if (target.size() != this->output_size)
    {
        throw std::runtime_error(std::format(
            "NeuralNetwork::backpropagate: Target size mismatch. Expected {}, got {}.",
            this->output_size, target.size()));
    }
    if (network_input.size() != this->input_size)
    {
        throw std::runtime_error(std::format(
            "NeuralNetwork::backpropagate: Network input size mismatch. Expected {}, got {}.",
            this->input_size, network_input.size()));
    }
    // IMPORTANT ASSUMPTION: feed_forward(network_input) must have been called *just before* this
    // to ensure the cached 'outputs' within each Layer and Neuron are correct for this input.

    // Iterate backwards through the layers
    for (std::size_t i = this->layers.size(); i-- > 0;)
    {
        Layer& current_layer = this->layers[i]; // Get non-const reference

        // Determine the layer *after* this one (in the forward pass)
        const Layer* next_layer_ptr = (i < this->layers.size() - 1) ? &this->layers[i + 1] : nullptr;

        // Determine the inputs that were fed INTO the current_layer during the forward pass
        std::span<const double> inputs_to_current_layer;
        std::vector<double>     prev_layer_outputs_storage; // Storage if needed

        if (i == 0)
        {
            // First layer (index 0) received the original network input
            inputs_to_current_layer = network_input;
        }
        else
        {
            // Hidden/Output layers received inputs from the previous layer's cached outputs
            // We need a copy because get_outputs() returns a const ref, and we need a stable span
            prev_layer_outputs_storage = this->layers[i - 1].get_outputs();
            inputs_to_current_layer    = prev_layer_outputs_storage;
        }

        // Perform backpropagation for the current layer
        // 'target' span is passed. Layer::backpropagate handles whether to use it (output layer)
        // or ignore it (hidden layer, using next_layer_ptr instead).
        current_layer.backpropagate(target, inputs_to_current_layer, learning_rate, next_layer_ptr);
    }
}


// Implementation for predict (single sample)
// Note: Removed 'const'
std::vector<double> NeuralNetwork::predict(std::span<const double> input)
{
    // Simply performs a forward pass
    return feed_forward(input);
}

// Implementation for predict (batch)
// Note: Removed 'const'
std::vector<std::vector<double>> NeuralNetwork::predict(const std::vector<std::vector<double>>& inputs)
{
    // Simply performs a forward pass for the batch
    return feed_forward(inputs);
}

// Implementation for compute_loss (static)
double NeuralNetwork::compute_loss(const std::vector<double>& output, const std::vector<double>& target)
{
    if (output.size() != target.size() || output.empty())
    {
        throw std::runtime_error(std::format(
            "NeuralNetwork::compute_loss: Output ({}) and target ({}) size mismatch or empty.",
            output.size(), target.size()));
    }

    double loss = 0.0;
    // Epsilon to prevent log(0)
    constexpr double epsilon = 1e-12; // Use a small epsilon

    for (std::size_t i = 0; i < output.size(); ++i)
    {
        // Clamp predicted value to avoid log(0) or log(1 - >1) issues
        const double predicted  = std::max(epsilon, std::min(1.0 - epsilon, output[i]));
        const double target_val = target[i]; // Assuming target is 0.0 or 1.0

        // Binary Cross-Entropy formula: - (y * log(p) + (1 - y) * log(1 - p))
        loss += -(target_val * std::log(predicted) + (1.0 - target_val) * std::log(1.0 - predicted));
    }

    // Return average loss across all outputs/labels for this sample
    return loss / static_cast<double>(output.size());
}


// Implementation for train
void NeuralNetwork::train(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets,
    const std::size_t                       epochs,
    const double                            learning_rate)
{
    if (inputs.size() != targets.size())
    {
        throw std::runtime_error(std::format(
            "NeuralNetwork::train: Number of input samples ({}) and target samples ({}) do not match.",
            inputs.size(), targets.size()));
    }
    if (inputs.empty())
    {
        std::cout << "NeuralNetwork::train: Warning - Training dataset is empty." << std::endl;
        return;
    }

    const std::size_t total_samples         = inputs.size();
    const std::size_t num_labels_per_sample = this->output_size; // Assuming targets[0] is valid if inputs not empty

    std::cout << "\n--- Starting Training ---" << std::endl;
    std::cout << "Dataset Size: " << total_samples << " samples" << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Learning Rate: " << learning_rate << std::endl;
    std::cout << "-------------------------" << std::endl;


    for (std::size_t epoch = 0; epoch < epochs; ++epoch)
    {
        auto        epoch_start                = std::chrono::high_resolution_clock::now();
        double      epoch_total_loss           = 0.0;
        std::size_t total_correct_labels_epoch = 0;

        // In a real scenario, you'd shuffle the data each epoch
        // std::vector<std::size_t> indices(total_samples);
        // std::iota(indices.begin(), indices.end(), 0);
        // std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
        // for (std::size_t idx : indices) { const auto& input = inputs[idx]; ... }

        for (std::size_t i = 0; i < total_samples; ++i)
        {
            // For now, process samples sequentially
            const auto& current_input  = inputs[i];
            const auto& current_target = targets[i];

            // Basic dimension checks (could be done once outside loop if guaranteed consistent)
            if (current_input.size() != this->input_size)
            {
                throw std::runtime_error(std::format("Train (Sample {}): Input size mismatch ({} vs {})", i,
                                                     current_input.size(), this->input_size));
            }
            if (current_target.size() != this->output_size)
            {
                throw std::runtime_error(std::format("Train (Sample {}): Target size mismatch ({} vs {})", i,
                                                     current_target.size(), this->output_size));
            }

            // 1. Forward Pass - Calculate predictions for the current input
            // Need std::span for feed_forward and backpropagate
            std::span<const double> input_span(current_input);
            std::vector<double>     output = feed_forward(input_span); // Non-const call

            // 2. Compute Loss for this sample
            epoch_total_loss += compute_loss(output, current_target);

            // 3. Backpropagation - Calculate gradients and update weights
            std::span<const double> target_span(current_target);
            backpropagate(input_span, target_span, learning_rate);

            // 4. (Optional) Calculate Accuracy for this sample
            for (std::size_t j = 0; j < num_labels_per_sample; ++j)
            {
                // Threshold prediction at 0.5 for binary classification per label
                const double predicted_label = (output[j] >= 0.5) ? 1.0 : 0.0;
                if (std::abs(predicted_label - current_target[j]) < 1e-6) // Compare floating points carefully
                {
                    total_correct_labels_epoch++;
                }
            }
        } // End loop through samples

        // Calculate average loss and accuracy for the epoch
        const double avg_epoch_loss = epoch_total_loss / static_cast<double>(total_samples);
        const double epoch_accuracy = static_cast<double>(total_correct_labels_epoch)
                                      / static_cast<double>(total_samples * num_labels_per_sample);

        auto                          epoch_end      = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;

        // Print epoch summary
        std::printf(
            "Epoch %*zu/%zu [%.3fs] - Avg Loss: %.6f | Accuracy: %.4f%%\n",
            static_cast<int>(std::to_string(epochs).length()), // Dynamic width for epoch number
            epoch + 1, epochs,
            epoch_duration.count(),
            avg_epoch_loss,
            epoch_accuracy * 100.0
        );
        // No \r needed if printing one line per epoch
        std::fflush(stdout); // Ensure output is displayed promptly
    }                        // End loop through epochs

    std::cout << "-------------------------" << std::endl;
    std::cout << "--- Training Finished ---" << std::endl;
}
