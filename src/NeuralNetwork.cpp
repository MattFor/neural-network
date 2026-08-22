//
// Created by MattFor on 12.04.2025.
//

#include "Utilities.h"
#include "NeuralNetwork.h"

#include <span>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>
#include <cstdio>
#include <format>
#include <random>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

/**
 * @brief Estimates the total number of trainable parameters (weights + biases) in a network topology.
 * @param layer_sizes Vector representing the number of neurons in each layer (input, hidden(s), output).
 * @return Estimated total number of parameters.
 */
static std::size_t estimate_parameters(const std::vector<std::size_t>& layer_sizes)
{
    if (layer_sizes.size() < 2)
    {
        return 0; // Need at least input and output layer
    }

    std::size_t total_params = 0;
    for (size_t i = 1; i < layer_sizes.size(); i++)
    {
        const std::size_t weights = layer_sizes[i - 1] * layer_sizes[i];
        const std::size_t biases  = layer_sizes[i]; // One bias per neuron in the current layer
        total_params              += weights + biases;
    }

    return total_params;
}

void NeuralNetwork::display_topology() const
{
    std::cout << "\n--- Neural Network Topology ---\n\n";
    std::cout << "Input Size: " << this->input_size << "\n";
    std::cout << "-----------------------------\n";

    // Define column widths for formatting
    constexpr int col_width_index      = 7;
    constexpr int col_width_neurons    = 9;
    constexpr int col_width_inputs     = 15;
    constexpr int col_width_type       = 12;
    constexpr int col_width_activation = 11; // New column width for activation function

    const std::string header_sep = " | ";
    const std::string row_sep    = "| ";

    // Print table header - Added Activation column
    std::cout << std::left << row_sep << std::setw(col_width_index) << "Layer #" << header_sep << std::setw(col_width_type) << "Type" << header_sep << std::setw(col_width_activation) << "Activation" // New Header
            << header_sep << std::setw(col_width_inputs) << "#Inputs/Neuron" << header_sep << std::setw(col_width_neurons) << "Neurons" << " |\n";

    // Print separator line based on calculated width - Adjusted for new column
    const auto total_width = col_width_index + col_width_type + col_width_activation + col_width_neurons + col_width_inputs + header_sep.length() * 4 // Adjusted: 4 separators
                             + row_sep.length() + 2;                                                                                                  // +2 for start/end |
    std::cout << std::string(total_width, '-') << "\n";

    std::cout << std::string(total_width, '-') << "\n";
    std::cout << std::left << row_sep << std::setw(col_width_index) << -1                                                   // Use -1 or a special index
            << header_sep << std::setw(col_width_type) << "Input" << header_sep << std::setw(col_width_activation) << "N/A" // Input layer typically has no activation
            << header_sep << std::setw(col_width_inputs) << "N/A"                                                           // Input layer has no preceding layer
            << header_sep << std::setw(col_width_neurons) << this->input_size << " |\n";
    std::cout << std::string(total_width, '-') << "\n";

    // Iterate through layers
    for (std::size_t i = 0; i < this->layers.size(); i++)
    {
        const auto&       layer                 = this->layers[i];
        // Assuming Layer class stores its neuron count directly or via neurons.size()
        const std::size_t num_neurons           = layer.get_neurons().size(); // Get neuron count
        // Input size for this layer is the neuron count of the previous layer (or network input size for the first layer)
        const std::size_t num_inputs_per_neuron = i == 0 ? this->input_size : this->layers[i - 1].get_neurons().size();

        // Determine layer type (Hidden or Output)
        std::string layer_type_str = i == this->layers.size() - 1 ? "Output" : "Hidden";

        // Get activation type string using the helper function
        std::string activation_str = activation_to_string(layer.get_activation_type());

        // Print layer information row - Added Activation string
        std::cout << std::left << row_sep << std::setw(col_width_index) << i << header_sep << std::setw(col_width_type) << layer_type_str << header_sep << std::setw(col_width_activation) << activation_str // New Column Data
                << header_sep << std::setw(col_width_inputs) << num_inputs_per_neuron << header_sep << std::setw(col_width_neurons) << num_neurons << " |\n";

        // Optional: Display sample weights/bias from the first neuron
        if (!layer.get_neurons().empty())
        {
            const auto&  neuron  = layer.get_neurons().front(); // Access first neuron
            const auto&  weights = neuron.get_weights();
            const double bias    = neuron.get_bias();

            // Print sample weights and bias, indented
            std::cout << row_sep << std::string(col_width_index, ' ') // Indentation
                    << " Sample W: [";
            // Print first few weights
            for (size_t w_idx = 0; w_idx < std::min(weights.size(), size_t{
                                                        3
                                                    }); w_idx++)
            {
                std::cout << std::fixed << std::setprecision(3) << weights[w_idx] << ( w_idx < 2 && w_idx < weights.size() - 1 ? ", " : "" );
            }

            if (weights.size() > 3)
            {
                std::cout << ", ...";
            } // Indicate more weights

            std::cout << "], Bias: " << std::fixed << std::setprecision(3) << bias << "\n";
        }

        // Print separator line after each layer's details
        std::cout << std::string(total_width, '-') << "\n";
    }

    // Print final output size
    std::cout << "Output Size: " << this->output_size << "\n";
    std::cout << "-----------------------------\n\n";
}

std::vector<double> NeuralNetwork::feed_forward(std::span<const double> input)
{
    if (input.size() != this->input_size)
    {
        throw std::runtime_error(std::format("NeuralNetwork::feed_forward: Input size mismatch. Expected {}, got {}.", this->input_size, input.size()));
    }

    std::vector current_activations(input.begin(), input.end());

    // Propagate through each layer.
    for (Layer& layer : this->layers)
    {
        const std::vector<double>& layer_output_ref = layer.feed_forward(current_activations);
        // Prepare for next iteration or final return
        current_activations                         = layer_output_ref; // Copy output for next layer's input or return
    }

    // The final activations after the last layer
    return current_activations;
}

std::vector<std::vector<double>> NeuralNetwork::feed_forward(const std::vector<std::vector<double>>& inputs)
{
    std::vector<std::vector<double>> results;
    results.reserve(inputs.size());

    // Process each sample individually
    for (const auto& sample : inputs)
    {
        // No need to check size here again if single feed_forward does it
        // std::span constructor performs no allocation
        results.push_back(feed_forward(std::span(sample)));
    }

    return results;
}

void NeuralNetwork::backpropagate(const std::span<const double> network_input, const std::span<const double> target, const double learning_rate)
{
    // Basic size checks
    if (target.size() != this->output_size)
    {
        throw std::runtime_error(std::format("NeuralNetwork::backpropagate: Target size mismatch. Expected {}, got {}.", this->output_size, target.size()));
    }

    if (network_input.size() != this->input_size)
    {
        throw std::runtime_error(std::format("NeuralNetwork::backpropagate: Network input size mismatch. Expected {}, got {}.", this->input_size, network_input.size()));
    }
    // IMPORTANT ASSUMPTION: feed_forward(network_input) must have been called *just before* this
    // to ensure the cached 'outputs' within each Layer and Neuron are correct for this input.

    if (this->layers.empty())
    {
        return;
    }

    // --- Pass 1; compute every delta walking backwards ---
    for (std::size_t i = this->layers.size(); i-- > 0;)
    {
        const Layer* next_layer_ptr = i + 1 < this->layers.size() ? &this->layers[i + 1] : nullptr;

        // 'target' is only consulted by the output layer; hidden layers derive their gradient from next_layer_ptr instead.
        this->layers[i].compute_gradients(target, next_layer_ptr, DEFAULT_GRADIENT_CLIP_THRESHOLD);
    }

    // --- Pass 2; apply the deltas ---
    // Now that all gradients are fixed, the weights can be updated in any order. Going
    // forwards lets each layer read the previous layer's cached activations directly,
    // with no copy: updating layer i does not disturb layer i-1's outputs.
    for (std::size_t i = 0; i < this->layers.size(); i++)
    {
        const std::span<const double> inputs_to_current_layer = i == 0 ? network_input : std::span(this->layers[i - 1].get_outputs());

        this->layers[i].apply_weight_updates(inputs_to_current_layer, learning_rate);
    }
}

std::vector<double> NeuralNetwork::predict(const std::span<const double> input)
{
    return feed_forward(input);
}


std::vector<std::vector<double>> NeuralNetwork::predict(const std::vector<std::vector<double>>& inputs)
{
    return feed_forward(inputs);
}

double NeuralNetwork::compute_loss(const std::vector<double>& output, const std::vector<double>& target)
{
    if (output.size() != target.size() || output.empty())
    {
        throw std::runtime_error(std::format("NeuralNetwork::compute_loss: Output ({}) and target ({}) size mismatch or empty.", output.size(), target.size()));
    }

    double loss = 0.0;

    for (std::size_t i = 0; i < output.size(); i++)
    {
        // Epsilon to prevent log(0)
        constexpr double epsilon    = 1e-12;
        // Clamp predicted value to avoid log(0) or log(1 - >1) issues
        const double     predicted  = std::max(epsilon, std::min(1.0 - epsilon, output[i]));
        const double     target_val = target[i]; // Assuming target is 0.0 or 1.0

        // Binary Cross-Entropy formula: - (y * log(p) + (1 - y) * log(1 - p))
        loss += -( target_val * std::log(predicted) + ( 1.0 - target_val ) * std::log(1.0 - predicted) );
    }

    // Return average loss across all outputs/labels for this sample
    return loss / static_cast<double>(output.size());
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, const std::size_t epochs, const double learning_rate)
{
    if (inputs.size() != targets.size())
    {
        throw std::runtime_error(std::format("NeuralNetwork::train: Number of input samples ({}) and target samples ({}) do not match.", inputs.size(), targets.size()));
    }

    if (inputs.empty())
    {
        std::cout << "NeuralNetwork::train: Warning - Training dataset is empty." << '\n';
        return;
    }

    const std::size_t total_samples = inputs.size();

    std::cout << "\n-------------------------" << '\n';
    std::cout << "Dataset Size: " << total_samples << " samples" << '\n';
    std::cout << "Epochs: " << epochs << '\n';
    std::cout << "Learning Rate: " << learning_rate << '\n';
    std::cout << "-------------------------" << '\n';


    // Create a vector to hold the indices of the samples
    std::vector<std::size_t> indices(total_samples);
    // Initialize it with 0, 1, 2, ..., total_samples - 1
    std::iota(indices.begin(), indices.end(), 0);

    // One engine for the whole run: re-seeding from std::random_device every epoch costs
    // entropy and time without making the shuffle any more random.
    std::mt19937 shuffle_engine{
        std::random_device{}()
    };

    for (std::size_t epoch = 0; epoch < epochs; epoch++)
    {
        auto        epoch_start         = std::chrono::high_resolution_clock::now();
        double      epoch_total_loss    = 0.0;
        std::size_t correct_predictions = 0;

        // --- Shuffle Data Indices for This Epoch ---
        // Shuffle the order in which samples are processed each epoch.
        // This helps the model generalize better and avoid getting stuck
        // due to the order of presentation.
        std::ranges::shuffle(indices, shuffle_engine);

        for (std::size_t i = 0; i < total_samples; i++)
        {
            // Visit the samples in the shuffled order, not in storage order.
            const std::size_t sample_index   = indices[i];
            const auto&       current_input  = inputs[sample_index];
            const auto&       current_target = targets[sample_index];

            // Basic dimension checks (could be done once outside loop if guaranteed consistent)
            if (current_input.size() != this->input_size)
            {
                throw std::runtime_error(std::format("Train (Sample {}): Input size mismatch ({} vs {})", sample_index, current_input.size(), this->input_size));
            }
            if (current_target.size() != this->output_size)
            {
                throw std::runtime_error(std::format("Train (Sample {}): Target size mismatch ({} vs {})", sample_index, current_target.size(), this->output_size));
            }

            // 1. Forward Pass - Calculate predictions for the current input
            // Need std::span for feed_forward and backpropagate
            const std::span     input_span(current_input);
            std::vector<double> output = feed_forward(input_span); // Non-const call

            // 2. Compute Loss for this sample
            epoch_total_loss += compute_loss(output, current_target);

            // 3. Backpropagation - Calculate gradients and update weights
            const std::span target_span(current_target);
            backpropagate(input_span, target_span, learning_rate);

            // 4. Calculate Accuracy for this sample
            // Single-label classification: the prediction is the highest-scoring output, and it
            // is correct when that index matches the target's. A per-label 0.5 threshold (as an
            // earlier version used) is badly misleading on one-hot data, because a network that outputs all zeros still scores (classes-1)/classes "accuracy".
            if (std::ranges::max_element(output) - output.begin() == std::ranges::max_element(current_target) - current_target.begin())
            {
                correct_predictions++;
            }
        } // End loop through samples

        // Calculate average loss and accuracy for the epoch
        const double avg_epoch_loss = epoch_total_loss / static_cast<double>(total_samples);
        const double epoch_accuracy = static_cast<double>(correct_predictions) / static_cast<double>(total_samples);

        auto                          epoch_end      = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;

        // Print epoch summary
        std::printf("\rEpoch %*zu/%zu [%.3fs] - Avg Loss: %.6f | Accuracy: %.4f%% ", static_cast<int>(std::to_string(epochs).length()), // Dynamic width for epoch number
                    epoch + 1, epochs, epoch_duration.count(), avg_epoch_loss, epoch_accuracy * 100.0);

        std::fflush(stdout); // Ensure output is displayed promptly
    }                        // End loop through epochs

    // The epoch progress line is redrawn in place with '\r'; close it before printing anything
    // else, or the banner below overwrites the final epoch's numbers.
    std::cout << '\n';
    std::cout << "-------------------------" << '\n';
    std::cout << "--- Training Finished ---" << '\n';
}


std::size_t NeuralNetwork::get_input_size() const noexcept
{
    return this->input_size;
}

std::size_t NeuralNetwork::get_output_size() const noexcept
{
    return this->output_size;
}

const std::vector<Layer>& NeuralNetwork::get_layers() const noexcept
{
    return this->layers;
}

std::vector<Layer>& NeuralNetwork::get_layers_mut() noexcept
{
    return this->layers;
}

const std::vector<std::size_t>& NeuralNetwork::get_layer_sizes() const noexcept
{
    return this->layer_sizes;
}

bool NeuralNetwork::load_weights(const std::string& filename)
{
    const std::filesystem::path path = resolve_project_path(std::filesystem::path("nn_data") / filename);

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << std::format("Could not open file for loading weights: {}", path.string()) << '\n';
        return false;
    }

    size_t loaded_input_size;
    size_t loaded_output_size;
    size_t num_layers;

    if (!file.read(reinterpret_cast<char*>(&loaded_input_size), sizeof( loaded_input_size )) || !file.read(reinterpret_cast<char*>(&loaded_output_size), sizeof( loaded_output_size )) || !file.read(reinterpret_cast<char*>(&num_layers), sizeof( num_layers )))
    {
        std::cerr << "Error reading network topology from file." << '\n';
        return false;
    }

    if (this->layers.size() != num_layers || this->input_size != loaded_input_size || this->output_size != loaded_output_size)
    {
        std::cerr << std::format("Network topology in file does not match current network (File: Input={}, Output={}, Layers={}; Current: Input={}, Output={}, Layers={}).", loaded_input_size, loaded_output_size, num_layers, this->input_size, this->output_size, this->layers.size()) << '\n';
        return false;
    }

    for (size_t i = 0; i < num_layers; i++)
    {
        ActivationType loaded_activation;
        size_t         num_neurons;
        if (!file.read(reinterpret_cast<char*>(&loaded_activation), sizeof( loaded_activation )) || !file.read(reinterpret_cast<char*>(&num_neurons), sizeof( num_neurons )))
        {
            std::cerr << std::format("Error reading layer {} info from file.", i) << '\n';
            return false;
        }

        if (this->layers[i].get_activation_type() != loaded_activation || this->layers[i].get_neurons().size() != num_neurons)
        {
            std::cerr << std::format("Layer {} configuration mismatch (File: Activation={}, Neurons={}; Current: Activation={}, Neurons={}).", i, static_cast<int>(loaded_activation), num_neurons, static_cast<int>(this->layers[i].get_activation_type()), this->layers[i].get_neurons().size()) << '\n';
            return false;
        }

        for (size_t j = 0; j < num_neurons; j++)
        {
            double bias;
            size_t num_weights;
            if (!file.read(reinterpret_cast<char*>(&bias), sizeof( bias )) || !file.read(reinterpret_cast<char*>(&num_weights), sizeof( num_weights )))
            {
                std::cerr << std::format("Error reading neuron {} in layer {} info from file.", j, i) << '\n';
                return false;
            }

            if (this->layers[i].get_neurons()[j].get_weights().size() != num_weights)
            {
                std::cerr << std::format("Neuron {} in layer {} weight size mismatch (File: {}; Current: {}).", j, i, num_weights, this->layers[i].get_neurons()[j].get_weights().size()) << '\n';
                return false;
            }

            double* weights_data_ptr = this->layers[i].get_neurons_mut()[j].get_weights_mut().data();

            // Calculate the total number of bytes needed as an unsigned type first
            const size_t total_bytes_unsigned = num_weights * sizeof(double);

            // Perform the read using the safely cast size
            if (const auto bytes_to_read = static_cast<std::streamsize>(total_bytes_unsigned); !file.read(reinterpret_cast<char*>(weights_data_ptr), bytes_to_read))
            {
                std::cerr << "Error reading weights for neuron " << j << " in layer " << i << " from file." << '\n';
                return false;
            }

            this->layers[i].get_neurons_mut()[j].set_bias(bias);
        }
    }

    return true;
}

void NeuralNetwork::save_weights(const std::string& filename) const
{
    const std::filesystem::path path = resolve_weights_directory() / filename;

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error(std::format("Could not open file for saving weights: {}", path.string()));
    }

    // Save network topology (layer sizes, activation types)
    const size_t num_layers = this->layers.size();
    file.write(reinterpret_cast<const char*>(&this->input_size), sizeof( this->input_size ));
    file.write(reinterpret_cast<const char*>(&this->output_size), sizeof( this->output_size ));
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof( num_layers ));
    for (const auto& layer : this->layers)
    {
        ActivationType hidden_activation = layer.get_activation_type();
        file.write(reinterpret_cast<const char*>(&hidden_activation), sizeof( hidden_activation ));
        size_t num_neurons = layer.get_neurons().size();
        file.write(reinterpret_cast<const char*>(&num_neurons), sizeof( num_neurons ));

        for (const auto& neuron : layer.get_neurons())
        {
            double bias        = neuron.get_bias();
            size_t num_weights = neuron.get_weights().size();
            file.write(reinterpret_cast<const char*>(&bias), sizeof( bias ));
            file.write(reinterpret_cast<const char*>(&num_weights), sizeof( num_weights ));
            file.write(reinterpret_cast<const char*>(neuron.get_weights().data()), static_cast<std::streamsize>(num_weights * sizeof(double)));
        }
    }

    file.close();

    if (!file)
    {
        throw std::runtime_error(std::format("Failed while writing weights to: {}", path.string()));
    }

    std::cout << std::format("Weights saved to: {}", path.string()) << '\n';
}

std::vector<std::size_t> NeuralNetwork::generate_layer_topology(const std::size_t num_input_features, std::size_t num_output_neurons, const std::size_t num_training_samples, const NetworkComplexity complexity_level)
{
    if (num_input_features == 0 || num_output_neurons == 0)
    {
        throw std::invalid_argument("Input and output feature counts must be greater than zero.");
    }

    std::vector<std::size_t> layer_sizes;
    layer_sizes.push_back(num_input_features); // Start with input layer

    // --- Heuristics ---

    // 1. Determine Number of Hidden Layers based on Complexity
    int num_hidden_layers = 0;
    switch (complexity_level)
    {
        case NetworkComplexity::SIMPLE:
        {
            num_hidden_layers = 1;
        }
        break;

        case NetworkComplexity::MEDIUM:
        {
            num_hidden_layers = 2;
        }
        break;

        case NetworkComplexity::COMPLEX:
        {
            num_hidden_layers = 3;
        }
        break; // Could be 2 or 3

        default:
        {
            num_hidden_layers = 2;
        }
        break;
    }

    // Very simple cases might only need 1 layer regardless of complexity setting
    if (num_input_features <= 10 && num_output_neurons <= 5)
    {
        if (complexity_level == NetworkComplexity::SIMPLE)
        {
            num_hidden_layers = 1;
        }
        else
        {
            num_hidden_layers = std::min(num_hidden_layers, 2); // Cap at 2 for simple I/O
        }
    }


    // 2. Determine Sizes of Hidden Layers
    // Base size heuristic - often between input and output size.
    // Using geometric mean sqrt(input * output) is one option, average is another.
    // Let's try a blend, leaning towards input size but capped.
    double base_hidden_size = std::sqrt(static_cast<double>(num_input_features) * static_cast<double>(num_output_neurons));
    base_hidden_size        = std::max({
        base_hidden_size,
        static_cast<double>(num_output_neurons * 2),
        16.0
    });
    // Ensure it's reasonably large, >= 2*output, min 16
    base_hidden_size = std::min(base_hidden_size, static_cast<double>(num_input_features * 2)); // Cap at 2*input

    // Adjust base size by complexity
    double complexity_scale = 1.0;
    switch (complexity_level)
    {
        case NetworkComplexity::SIMPLE:
        {
            complexity_scale = 0.6;
        }
        break; // Smaller for simple

        case NetworkComplexity::MEDIUM:
        {
            complexity_scale = 1.0;
        }
        break;

        case NetworkComplexity::COMPLEX:
        {
            complexity_scale = 1.5;
        }
        break; // Larger for complex
    }

    // If input/output are very different, adjust scale slightly
    if (const double io_ratio = static_cast<double>(num_input_features) / static_cast<double>(num_output_neurons); io_ratio > 10.0)
    {
        complexity_scale *= 1.1; // More capacity if reducing dimension significantly
    }
    else if (io_ratio < 0.1)
    {
        complexity_scale *= 0.9; // Less capacity if expanding dimension significantly
    }

    // Adjust base size *mildly* by data amount (avoid overfitting small data)
    // Using logarithmic scale or broad categories is better than linear.
    double data_amount_scale = 1.0;
    if (num_training_samples > 0)
    {
        // Avoid division by zero or log(0)
        if (num_training_samples < 1000)
        {
            data_amount_scale = 0.7; // Significantly reduce size for very small data
        }
        else if (num_training_samples < 10000)
        {
            data_amount_scale = 0.9; // Slightly reduce for small-medium data
        }
        else if (num_training_samples > 100000)
        {
            data_amount_scale = 1.2; // Slightly increase for large data
        }
        // Could use log10: data_amount_scale = 0.5 + 0.1 * std::log10(static_cast<double>(num_training_samples));
        // Needs clamping/tuning
    }


    // Calculate final size for the first hidden layer
    auto current_hidden_size = static_cast<std::size_t>(base_hidden_size * complexity_scale * data_amount_scale);

    // Ensure minimum hidden size
    current_hidden_size = std::max({
        current_hidden_size,
        static_cast<std::size_t>(8),
        num_output_neurons
    });


    // 3. Generate Hidden Layer Sizes (Pyramid/Funnel structure is common)
    double decay_factor = std::pow(static_cast<double>(num_output_neurons) / static_cast<double>(current_hidden_size), 1.0 / num_hidden_layers);
    decay_factor        = std::max(0.5, std::min(1.0, decay_factor)); // Keep decay reasonable (0.5 to 1.0)


    for (int i = 0; i < num_hidden_layers; i++)
    {
        // Ensure size doesn't drop below output size or a minimum threshold
        current_hidden_size = std::max({
            static_cast<std::size_t>(8),
            num_output_neurons,
            current_hidden_size
        });
        layer_sizes.push_back(current_hidden_size);

        // Decay size for the next layer (if not the last hidden layer)
        if (i < num_hidden_layers - 1)
        {
            // Option 1: Geometric Decay
            current_hidden_size = static_cast<std::size_t>(static_cast<double>(current_hidden_size) * decay_factor);

            // Option 2: Simpler Halving (can be too aggressive)
            // current_hidden_size /= 2;
        }
    }

    // --- Final Checks ---
    // Ensure last hidden layer isn't smaller than output (redundant due to loop check, but safe)
    if (layer_sizes.back() < num_output_neurons && layer_sizes.size() > 1)
    {
        layer_sizes.back() = std::max(num_output_neurons, static_cast<size_t>(8));
    }

    layer_sizes.push_back(num_output_neurons); // Add output layer

    // Optional: Parameter Count Check (Warning if parameters >> samples)
    if (const std::size_t estimated_param_count = estimate_parameters(layer_sizes); num_training_samples > 0 && estimated_param_count > num_training_samples * 5 && complexity_level != NetworkComplexity::COMPLEX)
    {
        // Scaling factor 5 is arbitrary - adjust as needed
        std::cout << "Warning: Generated topology has approx. " << estimated_param_count << " parameters, which might be high for " << num_training_samples << " training samples. Consider using NetworkComplexity::SIMPLE or more data.\n";
        // Could potentially auto-adjust here, but it gets complicated. Let's just warn.
    }
    else if (estimated_param_count > 0)
    {
        std::cout << "Info: Generated topology has approx. " << estimated_param_count << " parameters.\n";
    }

    return layer_sizes;
}
