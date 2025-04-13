// // main.cpp
// // Main entry point for the Neural Network application.
// // Created by MattFor on 13/04/2025. (Date corrected)
//
// #define DEV_MODE 0 // Set to 1 for development outputs (like displaying data)
//
// // Standard library includes not already in Utilities.h
// #include <vector>
// #include <string>
// #include <iostream>
// #include <random>    // For std::random_device, std::mt19937
// #include <stdexcept> // For std::exception
// #include <cstdlib>   // For EXIT_SUCCESS, EXIT_FAILURE
//
// // Local includes
// #include "include/NeuralNetwork.h"
// #include "include/Utilities.h" // Includes NeuralNetwork.h via Utilities.h/cpp
// // #include "transformations/transformations.h" // Include if needed (e.g., for data loading/preprocessing)
//
//
// int main()
// {
//     try
//     {
//         std::cout << "--- Neural Network Training & Evaluation ---\n" << std::endl;
//
//         // --- Configuration ---
//         // Simplified prompts as we removed Neuron mode and CSV loading for now
//         const std::size_t num_attributes = get_input<std::size_t>("Enter number of input attributes", 10);
//         const std::size_t num_targets = get_input<std::size_t>("Enter number of target classes (output neurons)", 5);
//         const std::size_t num_train_samples = get_input<std::size_t>("Enter number of training samples", 50000);
//         const std::size_t num_test_samples = get_input<std::size_t>("Enter number of test samples", 10000);
//         const std::size_t epochs = get_input<std::size_t>("Enter number of training epochs", 50);
//         const double      learning_rate = get_input<double>("Enter learning rate", 0.005);
//
//         // Define Network Topology (example: input -> hidden1 -> hidden2 -> output)
//         const std::vector<std::size_t> layer_sizes = {num_attributes, 128, 64, 32, num_targets};
//         // Input size is first element
//
//         // --- Data Generation ---
//         std::random_device rd;
//         std::mt19937       rng(rd());
//
//         std::vector<std::vector<double>> train_inputs,  test_inputs;
//         std::vector<std::vector<double>> train_targets, test_targets;
//
//         execute_timed(
//             "Generating training data...",
//             "Training data generated!",
//             generate_data_network,                           // Pass function directly
//             std::ref(train_inputs), std::ref(train_targets), // Pass vectors by reference
//             num_train_samples, num_attributes, num_targets, std::ref(rng)
//         );
//
//         execute_timed(
//             "Generating test data...",
//             "Test data generated!",
//             generate_data_network,
//             std::ref(test_inputs), std::ref(test_targets),
//             num_test_samples, num_attributes, num_targets, std::ref(rng)
//         );
//
//
//         if constexpr (DEV_MODE)
//         {
//             // Display a sample of the generated training data
//             display_train_data(train_inputs, train_targets, num_attributes);
//         }
//
//         // --- Network Creation ---
//         auto network = execute_timed(
//             "Creating Neural Network...",
//             "Neural Network created!",
//             // Lambda ensures construction happens within the timed block
//             [&]() -> NeuralNetwork // Specify return type for clarity
//             {
//                 return NeuralNetwork(layer_sizes); // Use defined topology
//             }
//         );
//
//         network.display_topology();
//
//         // --- Network Training ---
//         execute_timed(
//             "Starting Network Training...",
//             "Network Training finished!",
//             // Use a lambda to call the member function
//             [&]()
//             {
//                 network.train(train_inputs, train_targets, epochs, learning_rate);
//             }
//         );
//
//         // --- Network Evaluation ---
//         execute_timed(
//             "Starting Network Evaluation...",
//             "Network Evaluation finished!",
//             // Use a lambda to call the utility function
//             [&]()
//             {
//                 evaluate_network(network, test_inputs, test_targets);
//             }
//         );
//     } // End try block
//     catch (const std::exception& e)
//     {
//         std::cerr << "\n--- ERROR ---\n" << e.what() << "\n-------------" << std::endl;
//         std::cout << "Press ENTER to exit..." << std::endl;
//         std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Clear potential error state in cin
//         std::cin.get();
//         return EXIT_FAILURE;
//     }
//     catch (...) // Catch all other exceptions
//     {
//         std::cerr << "\n--- UNKNOWN ERROR ---\nAn unexpected error occurred." << std::endl;
//         std::cout << "Press ENTER to exit..." << std::endl;
//         std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
//         std::cin.get();
//         return EXIT_FAILURE;
//     }
//
//
//     std::cout << "\nExecution finished successfully." << std::endl;
//     std::cout << "Press ENTER to exit..." << std::endl;
//     // std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Might be needed if last input wasn't fully consumed
//     std::cin.get(); // Wait for user
//
//     return EXIT_SUCCESS;
// }

// main.cpp
// Main entry point for the Neural Network application.
// Created by MattFor on 13/04/2025. (Date corrected)

#define DEV_MODE 0 // Set to 1 for development outputs (like displaying data)

// Standard library includes not already in Utilities.h
#include <vector>
#include <string>
#include <iostream>
#include <random>    // For std::random_device, std::mt19937
#include <stdexcept> // For std::exception
#include <cstdlib>   // For EXIT_SUCCESS, EXIT_FAILURE

// Local includes
#include "include/NeuralNetwork.h"
#include "include/Utilities.h" // Includes NeuralNetwork.h via Utilities.h/cpp
// #include "transformations/transformations.h" // Include if needed (e.g., for data loading/preprocessing)


int main()
{
    try
    {
        std::cout << "--- Neural Network Training & Evaluation ---\n" << std::endl;

        // --- Load Data ---
        std::vector<std::vector<double>> all_inputs, all_targets;
        std::size_t num_attributes = 0;
        std::size_t num_targets = 0;

        execute_timed(
            "Loading data from zoo.csv...",
            "Data loaded!",
            load_zoo_data,
            "../data/zoo.csv", std::ref(all_inputs), std::ref(all_targets), std::ref(num_attributes), std::ref(num_targets)
        );

        if (all_inputs.empty()) {
            std::cerr << "Error: No data loaded. Exiting." << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Number of attributes loaded: " << num_attributes << std::endl;
        std::cout << "Number of target classes found: " << num_targets << std::endl;

        // --- Split Data into Training and Testing ---
        size_t split_index = static_cast<size_t>(0.7 * all_inputs.size()); // 70% for training
        std::vector<std::vector<double>> train_inputs(all_inputs.begin(), all_inputs.begin() + split_index);
        std::vector<std::vector<double>> train_targets(all_targets.begin(), all_targets.begin() + split_index);
        std::vector<std::vector<double>> test_inputs(all_inputs.begin() + split_index, all_inputs.end());
        std::vector<std::vector<double>> test_targets(all_targets.begin() + split_index, all_targets.end());

        std::cout << "Number of training samples: " << train_inputs.size() << std::endl;
        std::cout << "Number of test samples: " << test_inputs.size() << std::endl;

        // --- Configuration ---
        const std::size_t epochs = get_input<std::size_t>("Enter number of training epochs", 50);
        const double      learning_rate = get_input<double>("Enter learning rate", 0.005);

        // Define Network Topology
        std::vector<std::size_t> layer_sizes = {num_attributes, 64, 32, num_targets};
        // Input size is first element

        if constexpr (DEV_MODE)
        {
            // Display a sample of the loaded training data
            display_train_data(train_inputs, train_targets, num_attributes);
        }

        // --- Network Creation ---
        auto network = execute_timed(
            "Creating Neural Network...",
            "Neural Network created!",
            [&]() -> NeuralNetwork
            {
                return NeuralNetwork(layer_sizes);
            }
        );

        network.display_topology();

        // --- Network Training ---
        execute_timed(
            "Starting Network Training...",
            "Network Training finished!",
            [&]()
            {
                network.train(train_inputs, train_targets, epochs, learning_rate);
            }
        );

        // --- Network Evaluation ---
        execute_timed(
            "Starting Network Evaluation...",
            "Network Evaluation finished!",
            [&]()
            {
                evaluate_network(network, test_inputs, test_targets);
            }
        );
    } // End try block
    catch (const std::exception& e)
    {
        std::cerr << "\n--- ERROR ---\n" << e.what() << "\n-------------" << std::endl;
        std::cout << "Press ENTER to exit..." << std::endl;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
        return EXIT_FAILURE;
    }
    catch (...) // Catch all other exceptions
    {
        std::cerr << "\n--- UNKNOWN ERROR ---\nAn unexpected error occurred." << std::endl;
        std::cout << "Press ENTER to exit..." << std::endl;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
        return EXIT_FAILURE;
    }


    std::cout << "\nExecution finished successfully." << std::endl;
    std::cout << "Press ENTER to exit..." << std::endl;
    std::cin.get(); // Wait for user

    return EXIT_SUCCESS;
}