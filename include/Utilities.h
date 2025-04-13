// Utilities.h
// Helper functions for data handling, timing, and network evaluation.
// Created by MattFor on 13/04/2025. (Date corrected)

#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>  // Required for std::setw, std::left, std::fixed, std::setprecision
#include <stdexcept> // Required for std::runtime_error, std::invalid_argument
#include <limits>   // Required for std::numeric_limits
#include <type_traits> // Required for std::is_void_v
#include <utility>    // Required for std::forward
#include <algorithm>  // Required for std::max_element, std::find, std::min
#include <iterator>   // Required for std::distance
#include <cmath>      // Required for std::abs in evaluate_network
#include <cstdio>     // Required for std::printf, std::fflush
#include <functional>

// Forward declare NeuralNetwork to avoid including its full header here if possible
// (Though it might be needed for evaluate_network parameter type)
// #include "NeuralNetwork.h" // Include necessary headers if forward declaration isn't enough
class NeuralNetwork; // Forward declaration

void load_zoo_data(const std::string& filename, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets, std::size_t& num_attributes, std::size_t& num_targets);

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
    else
    {
        std::cout << "Invalid input. Using default value." << std::endl;
        return default_value;
    }
}


template <typename Func, typename... Args>
auto execute_timed(const std::string& startText, const std::string& endText, Func&& func, Args&&... args)
{
    std::cout << startText << std::endl; // Use endl to flush buffer
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
