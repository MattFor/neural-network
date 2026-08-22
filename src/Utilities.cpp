//
// Created by MattFor on 13.04.2025.
//

#include "Utilities.h"
#include "NeuralNetwork.h"

#include <ranges>
#include <numeric>
#include <algorithm>

namespace
{
    /// How far up the directory tree resolve_project_path() is willing to search.
    /// Deep enough for nested build trees such as `build/x64/Debug/`, shallow enough
    /// that a missing file does not turn into a walk to the filesystem root.
    constexpr int MAX_PARENT_SEARCH_DEPTH = 6;
}

std::filesystem::path resolve_project_path(const std::filesystem::path& relative_path)
{
    std::error_code       ec; // Keep the filesystem calls non-throwing; a failed probe just misses.
    std::filesystem::path directory = std::filesystem::current_path(ec);

    if (ec)
    {
        return relative_path;
    }

    for (int depth = 0; depth <= MAX_PARENT_SEARCH_DEPTH; depth++)
    {
        if (std::filesystem::path candidate = directory / relative_path; exists(candidate, ec))
        {
            return candidate;
        }

        const std::filesystem::path parent = directory.parent_path();
        if (parent.empty() || parent == directory) // Reached the filesystem root.
        {
            break;
        }

        directory = parent;
    }

    return relative_path;
}

std::filesystem::path resolve_weights_directory()
{
    std::error_code ec;

    if (std::filesystem::path existing = resolve_project_path("nn_data"); is_directory(existing, ec))
    {
        return existing;
    }

    std::filesystem::path parent = resolve_project_path("datasets").parent_path();
    if (parent.empty())
    {
        parent = std::filesystem::current_path(ec);
        if (ec)
        {
            parent = ".";
        }
    }

    std::filesystem::path target = parent / "nn_data";
    create_directories(target, ec);

    return ec
               ? std::filesystem::path{
                   "nn_data"
               }
               : target;
}

// Display multi-label training data
void display_train_data(const std::vector<std::vector<double>>& train_inputs, const std::vector<std::vector<double>>& train_targets, const std::size_t num_attributes)
{
    if (train_inputs.empty() || train_targets.empty())
    {
        std::cout << "No training data to display." << '\n';
        return;
    }

    if (train_inputs.size() != train_targets.size())
    {
        std::cerr << "Warning: Mismatch between number of input samples and target samples in display_train_data." << '\n';
        return;
    }

    // Determine the approximate width needed for a feature display
    std::size_t feature_width = num_attributes * 6; // Estimate width based on the number format below
    feature_width             = std::max(feature_width, std::size_t{
                                             15
                                         }); // Minimum width

    std::cout << "\n--- Sample Training Data ---\n";
    // Print headers
    std::cout << std::left << std::setw(static_cast<int>(feature_width)) << "Input Features" << " | " << "Target Vector (One-Hot)" << '\n';
    std::cout << std::string(feature_width + 28, '-') << '\n'; // Adjust separator width

    // Print a few rows of training data (e.g. first 5 and last 5)
    constexpr std::size_t display_count = 5;
    for (std::size_t i = 0; i < std::min(train_inputs.size(), display_count); i++)
    {
        // Print feature vector
        std::cout << std::left << std::setw(static_cast<int>(feature_width));
        std::stringstream ss_features;
        for (std::size_t j = 0; j < train_inputs[i].size(); j++)
        {
            ss_features << std::fixed << std::setprecision(2) << train_inputs[i][j] << ( j < train_inputs[i].size() - 1 ? " " : "" );
        }
        std::cout << ss_features.str();

        std::cout << " | ";

        // Print target vector
        if (i < train_targets.size())
        {
            // Check bounds for targets as well
            for (const auto& val : train_targets[i])
            {
                std::cout << static_cast<int>(val) << " "; // Print targets as 0 or 1
            }
        }
        std::cout << '\n';
    }

    if (train_inputs.size() > display_count * 2)
    {
        std::cout << std::left << std::setw(static_cast<int>(feature_width)) << "..." << " | ..." << '\n';
        for (std::size_t i = train_inputs.size() - display_count; i < train_inputs.size(); i++)
        {
            // Print feature vector
            std::cout << std::left << std::setw(static_cast<int>(feature_width));
            std::stringstream ss_features;

            for (std::size_t j = 0; j < train_inputs[i].size(); j++)
            {
                ss_features << std::fixed << std::setprecision(2) << train_inputs[i][j] << ( j < train_inputs[i].size() - 1 ? " " : "" );
            }

            std::cout << ss_features.str();
            std::cout << " | ";

            // Print target vector
            if (i < train_targets.size())
            {
                for (const auto& val : train_targets[i])
                {
                    std::cout << static_cast<int>(val) << " ";
                }
            }
            std::cout << '\n';
        }
    }
    else if (train_inputs.size() > display_count)
    {
        // Print remaining if fewer than 2*display_count but more than display_count
        for (std::size_t i = display_count; i < train_inputs.size(); i++)
        {
            // Print feature vector
            std::cout << std::left << std::setw(static_cast<int>(feature_width));
            std::stringstream ss_features;

            for (std::size_t j = 0; j < train_inputs[i].size(); j++)
            {
                ss_features << std::fixed << std::setprecision(2) << train_inputs[i][j] << ( j < train_inputs[i].size() - 1 ? " " : "" );
            }

            std::cout << ss_features.str();
            std::cout << " | ";

            // Print target vector
            if (i < train_targets.size())
            {
                for (const auto& val : train_targets[i])
                {
                    std::cout << static_cast<int>(val) << " ";
                }
            }

            std::cout << '\n';
        }
    }

    std::cout << std::string(feature_width + 28, '-') << '\n';
    std::cout << '\n';
}


// Generate synthetic data for network (finding max input index)
void generate_data_network(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets, const std::size_t num_samples, const std::size_t num_attributes, const std::size_t num_targets, std::mt19937& rng)
{
    std::uniform_real_distribution input_dist(0.0, 1.0); // Inputs [0, 1]

    inputs.resize(num_samples, std::vector<double>(num_attributes));
    targets.resize(num_samples, std::vector(num_targets, 0.0)); // Initialise targets with zeros

    for (std::size_t i = 0; i < num_samples; i++)
    {
        // Generate random inputs
        for (std::size_t j = 0; j < num_attributes; j++)
        {
            inputs[i][j] = input_dist(rng);
        }

        // Find the index of the largest input value
        // Use iterators for safety and generality

        if (auto max_it = std::ranges::max_element(inputs[i]); max_it != inputs[i].end())
        // Check if the input vector was not empty
        {
            // Ensure the index is valid for the target vector size
            if (const std::size_t target_class_index = std::distance(inputs[i].begin(), max_it); target_class_index < num_targets)
            {
                targets[i][target_class_index] = 1.0; // Set the one-hot target
            }
            else
            {
                // This case should ideally not happen if num_targets >= num_attributes
                // or if the task definition changes. Handle gracefully.
                // std::cerr << "Warning: Max input index (" << target_class_index
                //         << ") exceeds the number of target classes (" << num_targets
                //         << ") for sample " << i << ". Setting target to class 0." << '\n';
                if (!targets[i].empty())
                {
                    targets[i][0] = 1.0;
                }
            }
        }
        else if (!targets[i].empty()) // Handle case of an empty input vector
        {
            std::cerr << "Warning: Empty input vector generated for sample " << i << ". Setting target to class 0." << '\n';
            targets[i][0] = 1.0; // Default target
        }
    }

    std::cout << "Generated " << num_samples << " samples for network training/testing." << '\n';
}


// Evaluate the network using classification accuracy for one-hot encoded targets
void evaluate_network(NeuralNetwork& network, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets)
{
    if (inputs.empty())
    {
        std::printf("Evaluation: No test data provided.\n");
        return;
    }

    if (inputs.size() != targets.size())
    {
        throw std::runtime_error("Evaluation Error: Input and target counts mismatch.");
    }

    std::size_t       correct_predictions = 0;
    double            total_loss          = 0.0;
    const std::size_t num_samples         = inputs.size();

    for (std::size_t i = 0; i < num_samples; i++)
    {
        // Use std::span for prediction
        const std::span     input_span(inputs[i]);
        // Call the const predict method
        std::vector<double> output_vec = network.predict(input_span);

        const auto& target_vec = targets[i];

        if (output_vec.size() != target_vec.size())
        {
            throw std::runtime_error(std::format("Evaluation Error: Output size ({}) mismatch with target size ({}) at index {}.", output_vec.size(), target_vec.size(), i));
        }

        if (target_vec.empty())
        {
            std::cerr << "Warning: Empty target vector encountered during evaluation at index " << i << '\n';
            continue; // Skip this sample
        }

        // --- Accuracy Calculation (for Classification) ---
        // Find predicted class: index of the maximum output value
        const auto predicted_it = std::ranges::max_element(output_vec);

        // Find the actual class the same way, rather than searching for an exact 1.0. That
        // makes the check independent of the exact encoding (hard one-hot, smoothed labels,
        // or values that a lossy round trip nudged off 1.0).

        if (const auto actual_it = std::ranges::max_element(target_vec); *actual_it <= 0.0)
        {
            // Every entry is zero/negative: there is no class to be right about.
            std::cerr << "Warning: Target vector for sample " << i << " has no positive entry. Check data generation/loading." << '\n';
        }
        else if (std::distance(output_vec.begin(), predicted_it) == std::distance(target_vec.begin(), actual_it))
        {
            correct_predictions++;
        }

        // --- Loss Calculation ---
        try
        {
            total_loss += NeuralNetwork::compute_loss(output_vec, target_vec);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Warning: Error computing loss for sample " << i << ": " << e.what() << '\n';
            // Optionally skip the sample or assign max loss
        }
    } // End loop through samples

    // Calculate overall average loss and accuracy
    const double avg_loss = num_samples == 0 ? 0.0 : total_loss / static_cast<double>(num_samples);
    const double accuracy = num_samples == 0 ? 0.0 : static_cast<double>(correct_predictions) / static_cast<double>(num_samples);

    // Print the evaluation results
    std::printf("--- Evaluation Results ---\n");
    std::printf(" Average Loss: %.8f\n", avg_loss);
    std::printf(" Accuracy:     %.4f%% (%zu / %zu correct)\n", accuracy * 100.0, correct_predictions, num_samples);
    std::printf("--------------------------\n");
    std::fflush(stdout);
}

/**
 * @brief Implementation of the data splitting function.
 * (See Utilities.h for detailed parameter and return value documentation)
 */
SplitDataResult split_data(const std::vector<std::vector<double>>& all_inputs,  // Input features dataset
                           const std::vector<std::vector<double>>& all_targets, // Corresponding target outputs dataset
                           const double                            train_ratio, // Proportion for training set (0.0 to 1.0)
                           const bool                              shuffle      // Whether to shuffle before splitting
)
{
    // --- Input Validation ---

    // Check if input and target vectors have the same number of samples.
    if (all_inputs.size() != all_targets.size())
    {
        throw std::invalid_argument("split_data Error: Input features count (" + std::to_string(all_inputs.size()) + ") does not match target outputs count (" + std::to_string(all_targets.size()) + ").");
    }

    // Check if the data are empty.
    if (all_inputs.empty()) // No need to check all_targets again due to the previous check
    {
        throw std::invalid_argument("split_data Error: Cannot split empty data.");
    }

    // Check if the training ratio is within the valid range (exclusive of 0 and 1).
    // A ratio of 0 or 1 would result in one empty set, which might be undesirable.
    if (train_ratio <= 0.0 || train_ratio >= 1.0)
    {
        throw std::invalid_argument("split_data Error: Training ratio must be between 0.0 and 1.0 (exclusive). Received: " + std::to_string(train_ratio));
    }

    // --- Index Preparation and Shuffling ---

    // Get the total number of samples in the dataset.
    const std::size_t total_samples = all_inputs.size();

    // Create a vector of indices: [0, 1, 2, ..., total_samples - 1].
    std::vector<std::size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0); // Efficiently fills the vector with sequential numbers.

    // Shuffle the indices if requested. This randomises the order of samples
    // before splitting, ensuring unbiased train/test sets.
    if (shuffle)
    {
        // Use a random device to seed the Mersenne Twister engine for better randomness.
        std::random_device rd;
        // Mersenne Twister 19937 generator: a high-quality pseudo-random number generator.
        std::mt19937       random_engine(rd());
        // Perform the in-place shuffle on the indices vector using the random engine.
        std::ranges::shuffle(indices, random_engine);
    }

    // --- Calculate Split Point ---

    // Calculate the number of samples for the training set based on the ratio.
    // Cast total_samples to double for the multiplication, then cast the result back to size_t.
    // Using std::round before casting can sometimes be slightly more robust against floating point inaccuracies,
    // although static_cast is usually enough here.
    // Const auto train_count = static_cast<std::size_t>(std::round(static_cast<double>(total_samples) * train_ratio));
    const auto train_count = static_cast<std::size_t>(static_cast<double>(total_samples) * train_ratio);

    // Calculate the number of samples for the testing set.
    // This ensures that train_count + test_count == total_samples.
    const std::size_t test_count = total_samples - train_count;

    // Basic sanity check: ensure neither set is empty after calculation (due to rounding or extreme ratios near 0/1)
    // This check might be redundant given the train_ratio validation, but adds safety.
    if (train_count == 0 || test_count == 0)
    {
        throw std::runtime_error( // Use runtime_error for an unexpected internal state
            "split_data Error: Calculated split results in an empty training or testing set. Train count: " + std::to_string(train_count) + ", Test count: " + std::to_string(test_count));
    }

    // --- Data Allocation and Copying ---

    // Create the DataPair objects to hold the split data.
    DataPair train_data; // Will hold {training_inputs, training_targets}
    DataPair test_data;  // Will hold {testing_inputs, testing_targets}

    // Reserve memory in the vectors to avoid multiple reallocations during push_back,
    // improving performance.
    train_data.first.reserve(train_count);  // Reserve space for training inputs
    train_data.second.reserve(train_count); // Reserve space for training targets
    test_data.first.reserve(test_count);    // Reserve space for testing inputs
    test_data.second.reserve(test_count);   // Reserve space for testing targets

    // Populate the training data vectors using the first 'train_count' shuffled indices.
    for (std::size_t i = 0; i < train_count; i++)
    {
        // Get the original index of the sample from the shuffled indices vector.
        const std::size_t original_index = indices[i];
        // Add the corresponding input sample to the training inputs vector.
        train_data.first.push_back(all_inputs[original_index]);
        // Add the corresponding target sample to the training targets vector.
        train_data.second.push_back(all_targets[original_index]);
    }

    // Populate the testing data vectors using the remaining shuffled indices.
    for (std::size_t i = train_count; i < total_samples; i++)
    {
        // Get the original index of the sample from the shuffled indices vector.
        const std::size_t original_index = indices[i];
        // Add the corresponding input sample to the testing inputs vector.
        test_data.first.push_back(all_inputs[original_index]);
        // Add the corresponding target sample to the testing targets vector.
        test_data.second.push_back(all_targets[original_index]);
    }

    // --- Return Result ---

    // Return the final result as a pair containing the training DataPair and the testing DataPair.
    // This structure { {train_in, train_out}, {test_in, test_out} } matches the SplitDataResult type.
    return {
        train_data,
        test_data
    };
}

/**
 * @brief Implementation of the function to display sample data points.
 * (See Utilities.h for detailed parameter and purpose documentation)
 */
void display_sample_data(const std::string&                      title,   // Title for the output block
                         const std::vector<std::vector<double>>& inputs,  // Input dataset
                         const std::vector<std::vector<double>>& targets, // Target dataset
                         const std::size_t                       count)   // Max number of samples to show
{
    // --- Print Title ---
    // Provides context for the printed data block.
    std::cout << "\n--- " << title << " ---" << '\n';

    // --- Basic Input Validation ---

    // Check if the requested display count is zero.
    if (count == 0)
    {
        std::cout << "Display count is zero. No samples will be shown." << '\n';
        std::cout << "--- End of Sample Data ---" << '\n';
        return; // Nothing more to do.
    }

    // Check if the input data container is empty.
    if (inputs.empty())
    {
        std::cout << "Input data is empty. Nothing to display." << '\n';
        std::cout << "--- End of Sample Data ---" << '\n';
        return;
    }

    // Check if the target data container is empty.
    if (targets.empty())
    {
        std::cout << "Target data is empty. Nothing to display." << '\n';
        std::cout << "--- End of Sample Data ---" << '\n';
        return;
    }

    // Check if the number of input samples matches the number of target samples.
    // This is crucial for displaying meaningful input-target pairs.
    if (inputs.size() != targets.size())
    {
        // Output a warning if sizes mismatch, as pairs cannot be reliably formed.
        std::cout << "[Warning] display_sample_data: Input size (" << inputs.size() << ") and Target size (" << targets.size() << ") mismatch. Cannot display pairs reliably." << '\n';
        std::cout << "--- End of Sample Data ---" << '\n';
        // It's safer to return here than to potentially access out-of-bounds data
        // or display misleading pairs.
        return;
    }

    // --- Determine Actual Number of Samples to Display ---
    // Display up to 'count' samples or fewer if the dataset size is smaller.
    const std::size_t num_samples_to_display = std::min(count, inputs.size());

    // Inform the user how many samples are actually being shown.
    std::cout << "Displaying first " << num_samples_to_display << " of " << inputs.size() << " total samples:" << '\n';

    // --- Set Output Formatting ---
    // Use fixed-point notation with specific precision for consistent alignment
    // and readability of floating-point numbers.
    std::cout << std::fixed << std::setprecision(4); // Example: 4 decimal places

    // --- Iterate and Print Samples ---
    // Loop through the determined number of samples.
    for (std::size_t i = 0; i < num_samples_to_display; i++)
    {
        // Print a header for each sample.
        std::cout << "--- Sample " << i + 1 << " ---" << '\n';

        // --- Display Input Vector ---
        std::cout << "  Input:  [";
        // Check if the specific input sample vector is not empty.
        if (!inputs[i].empty())
        {
            // Iterate through the features (doubles) within the input sample vector.
            for (std::size_t j = 0; j < inputs[i].size(); j++)
            {
                // Print the feature value.
                std::cout << inputs[i][j];
                // Print a comma and space unless it's the last element.
                if (j < inputs[i].size() - 1)
                {
                    std::cout << ", ";
                }
            }
        }
        else
        {
            // Indicate if a specific input sample vector is empty.
            std::cout << "(empty)";
        }
        std::cout << "]" << '\n'; // Close bracket and add newline

        // --- Display Target Vector ---
        std::cout << "  Target: [";
        // Check if the specific target sample vector is not empty.
        if (!targets[i].empty())
        {
            // Iterate through the values within the target sample vector.
            for (std::size_t j = 0; j < targets[i].size(); j++)
            {
                // Print the target value.
                std::cout << targets[i][j];
                // Print a comma and space unless it's the last element.
                if (j < targets[i].size() - 1)
                {
                    std::cout << ", ";
                }
            }
        }
        else
        {
            // Indicate if a specific target sample vector is empty.
            std::cout << "(empty)";
        }

        std::cout << "]" << '\n'; // Close bracket and add newline
    }

    // --- Reset Output Formatting (Good Practice) ---
    // Reset the iostream flags to their default state to avoid affecting
    // later output operations elsewhere in the program.
    std::cout.unsetf(std::ios_base::floatfield); // Remove the fixed / scientific flag
    std::cout << std::setprecision(6);           // Reset precision (default is often 6)

    // --- Print Footer ---
    std::cout << "--- End of Sample Data ---" << '\n';
}
