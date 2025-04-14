//
// Created by MattFor on 12.04.2025.
//

// Utilities.h
// Helper functions for data handling, timing, and network evaluation.
// Created by MattFor on 13/04/2025. (Date corrected)

#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cstdio>      // Required for std::printf, std::fflush
#include <fstream>
#include <sstream>
#include <utility>     // Required for std::forward
#include <iostream>
#include <functional>
#include <type_traits> // Required for std::is_void_v

// Represents a pair of input vectors and corresponding target vectors.
// Format: { vector_of_input_samples, vector_of_target_samples }
using DataPair = std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>;

// Represents the result of splitting data into training and testing sets.
// Format: { training_DataPair, testing_DataPair }
// which expands to: { {train_inputs, train_targets}, {test_inputs, test_targets} }
using SplitDataResult = std::pair<DataPair, DataPair>;

// We want to store the function related to the data file.
using DataProcessingFunc = std::function<DataPair()>;

// Forward declare NeuralNetwork to avoid including its full header here if possible
// (Though it might be needed for evaluate_network parameter type)
// #include "NeuralNetwork.h" // Include necessary headers if forward declaration isn't enough
class NeuralNetwork; // Forward declaration

// --- Function Declarations ---

/**
 * @brief Gets user input with a prompt and default value.
 * @tparam T The type of input expected.
 * @param prompt The message displayed to the user.
 * @param default_value The value returned if input is empty or invalid.
 * @return The user's input or the default value.
 */
template <typename T>
T get_input(const std::string& prompt, T default_value);

/**
 * @brief Executes a function and prints the elapsed time.
 * @tparam Func Callable function type.
 * @tparam Args Argument types for the function.
 * @param startText Message printed before execution.
 * @param endText Message printed after execution.
 * @param func The function to execute.
 * @param args Arguments to pass to the function.
 * @return The result of the function if it's non-void.
 */
template <typename Func, typename... Args>
auto execute_timed(const std::string& startText, const std::string& endText, Func&& func, Args&&... args);

/**
 * @brief Displays multi-label training data (features and target vectors).
 * @param train_inputs Vector of input feature vectors.
 * @param train_targets Vector of target output vectors (e.g., one-hot).
 * @param num_attributes Expected number of features per input sample (for formatting).
 */
void display_train_data(
    const std::vector<std::vector<double>>& train_inputs,
    const std::vector<std::vector<double>>& train_targets,
    std::size_t                             num_attributes
);

/**
 * @brief Generates synthetic data for network training (multi-label/classification).
 * Creates inputs and corresponding one-hot encoded targets where the '1'
 * indicates the index of the largest value in the input vector.
 * @param inputs Output vector to store generated input features.
 * @param targets Output vector to store generated target vectors (one-hot).
 * @param num_samples Number of samples to generate.
 * @param num_attributes Number of features per input sample.
 * @param num_targets Number of possible target classes (size of the one-hot vector).
 * @param rng Mersenne Twister random number generator.
 */
void generate_data_network(
    std::vector<std::vector<double>>& inputs,
    std::vector<std::vector<double>>& targets,
    std::size_t                       num_samples,
    std::size_t                       num_attributes,
    std::size_t                       num_targets,
    std::mt19937&                     rng
);

/**
 * @brief Evaluates the network's performance on a test dataset (classification accuracy).
 * Calculates accuracy based on comparing the index of the highest output activation
 * with the index of the '1' in the one-hot encoded target vector.
 * @param network The trained neural network (const reference).
 * @param inputs The input feature vectors of the test set.
 * @param targets The corresponding one-hot encoded target vectors of the test set.
 * @throws std::runtime_error If input/target counts or sizes mismatch.
 */
void evaluate_network(
    NeuralNetwork&                          network, // Now const
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets
);

/**
 * @brief Splits input and target datasets into training and testing subsets.
 *
 * Takes complete datasets of input features and corresponding target outputs,
 * shuffles them optionally, and then divides them into two distinct sets
 * (training and testing) based on the specified ratio.
 *
 * @param all_inputs A constant reference to the vector containing all input samples.
 * Each inner vector represents a single input sample's features.
 * @param all_targets A constant reference to the vector containing all target samples.
 * Each inner vector represents a single target sample's outputs.
 * Must have the same number of samples as all_inputs.
 * @param train_ratio The desired proportion of the dataset to allocate to the
 * training set. Must be a value strictly between 0.0 and 1.0
 * (e.g., 0.8 means 80% for training, 20% for testing).
 * @param shuffle Optional boolean flag (defaults to true). If true, the dataset
 * indices are randomly shuffled before splitting, ensuring that
 * training and testing sets are randomized samples. If false,
 * the split occurs sequentially (first N samples for training).
 *
 * @return A SplitDataResult object (std::pair<DataPair, DataPair>).
 * The first element of the pair is the training set (itself a DataPair
 * containing training inputs and training targets).
 * The second element of the pair is the testing set (a DataPair
 * containing testing inputs and testing targets).
 * Example: return_value.first.first contains training inputs.
 * return_value.second.second contains testing targets.
 *
 * @throws std::invalid_argument if:
 * - all_inputs and all_targets have different sizes.
 * - all_inputs (and therefore all_targets) is empty.
 * - train_ratio is not strictly between 0.0 and 1.0.
 */
SplitDataResult split_data(
    const std::vector<std::vector<double>>& all_inputs,
    const std::vector<std::vector<double>>& all_targets,
    double                                  train_ratio,
    bool                                    shuffle = true
);

/**
 * @brief Displays a specified number of sample data points (input/target pairs).
 *
 * Prints the first 'count' samples from the provided input and target datasets
 * to the standard output stream in a formatted way. Handles cases where the dataset
 * is smaller than 'count', or if the data is empty or mismatched. Useful for
 * debugging or inspecting generated/loaded data.
 *
 * @param title A descriptive title to print before the sample data block.
 * @param inputs A constant reference to the vector containing all input samples.
 * Each inner vector represents a single sample's features.
 * @param targets A constant reference to the vector containing all target samples.
 * Each inner vector represents a single sample's outputs. Must correspond
 * 1:1 with the samples in 'inputs'.
 * @param count The maximum number of samples to display. If the dataset contains
 * fewer samples than 'count', all available samples will be displayed.
 * If count is 0, nothing is displayed.
 */
void display_sample_data(
    const std::string&                      title,
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets,
    std::size_t                             count
);

// --- Template Implementations (must be in header) ---

template <typename T>
T get_input(const std::string& prompt, T default_value)
{
    std::cout << prompt << " (default: " << default_value << "): ";
    std::string input_line;
    std::getline(std::cin, input_line);

    if (input_line.empty())
    {
        return default_value;
    }

    std::istringstream stream(input_line);
    T                  value;

    // Check if extraction is successful and if the entire stream was consumed (optional, for stricter input)
    if (stream >> value && stream.eof())
    {
        return value;
    }

    std::cout << "Invalid input. Using default value." << '\n';
    return default_value;
}


template <typename Func, typename... Args>
auto execute_timed(const std::string& startText, const std::string& endText, Func&& func, Args&&... args)
{
    std::cout << startText << '\n';
    const auto start_time = std::chrono::high_resolution_clock::now();

    // Using if constexpr to handle both void and non-void functions.
    if constexpr (std::is_void_v<std::invoke_result_t<Func, Args...>>) // C++17 invoke_result_t
    {
        // Call the function with forwarded arguments
        std::invoke(std::forward<Func>(func), std::forward<Args>(args)...); // C++17 invoke

        const auto                          end_time = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> duration = end_time - start_time;
        std::printf("%s Took: %.6f seconds\n", endText.c_str(), duration.count());
        std::fflush(stdout);
        // No return value for void functions
    }
    else
    {
        // Call the function and store the result
        auto result = std::invoke(std::forward<Func>(func), std::forward<Args>(args)...); // C++17 invoke

        const auto                          end_time = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> duration = end_time - start_time;
        std::printf("%s Took: %.6f seconds\n", endText.c_str(), duration.count());
        std::fflush(stdout);

        return result; // Return the result
    }
}


#endif // UTILITIES_H
