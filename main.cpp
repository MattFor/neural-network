//
// Created by MattFor on 12.04.2025.
//

#define DEV_MODE 0 // Set to 1 for development outputs

#include <map>
#include <ranges>
#include <vector>
#include <string>
#include <random>           // For std::random_device, std::mt19937
#include <cstdlib>          // For EXIT_SUCCESS, EXIT_FAILURE
#include <utility>          // For std::pair
#include <iostream>
#include <filesystem>       // For checking file existence (C++17)
#include <functional>       // For std::function
#include <unordered_map>    // For dataset processor map

#include "include/Utilities.h"
#include "include/NeuralNetwork.h"
#include "data/DataTransformations.h"

// --- Main Application Logic ---
int main()
{
    try // Top-level error handling
    {
        std::cout << "--- Neural Network by MattFor ---\n" << '\n';

        // --- Dataset Processors Map ---
        // Maps dataset identifiers (e.g., filenames) to their processing functions.
        // Easily extendable with more datasets.
        const std::unordered_map<std::string, DataProcessingFunc> dataset_processors =
        {
            {"car.data", process_car},
            {"mnist", process_mnist}
        };

        bool running = true;

        while (running)
        {
            // --- Dataset Selection ---
            std::cout << "Available datasets:" << '\n';
            for (const auto& name : std::ranges::views::keys(dataset_processors))
            {
                std::cout << " - " << name << '\n';
            }

            std::cout << '\n';

            auto dataset_choice = get_input<std::string>("Enter dataset name", "Randomly Generated Data");

            // --- Data Storage ---
            // Declared here to be accessible for training/evaluation later in the chosen branch
            std::size_t                      num_targets    = 0;
            std::size_t                      num_attributes = 0;
            std::vector<std::size_t>         layer_sizes;
            std::vector<std::vector<double>> train_inputs,  test_inputs;
            std::vector<std::vector<double>> train_targets, test_targets;

            // --- Configuration (Defaults adjusted) ---
            std::size_t      epochs            = 100;  // More epochs often needed
            double           learning_rate     = 0.01; // Common starting point
            constexpr double train_split_ratio = 0.8;  // For splitting existing datasets

            // --- Processing based on choice ---
            if (const auto processor_iter = dataset_processors.find(dataset_choice);
                processor_iter != dataset_processors.end()) // Existing dataset chosen
            {
                std::cout << "\nProcessing dataset: " << dataset_choice << "..." << '\n';

                // Check if data file actually exists (basic check)
                if (!std::filesystem::exists(dataset_choice))
                {
                    std::cerr << "Warning: Data file '" << dataset_choice
                            << "' not found in the execution directory. Processing might fail." << '\n';
                    // Depending on the processor function, it might handle path internally or fail now.
                }

                // Use the mapped function to get all data
                DataPair all_data;
                execute_timed(
                    "Loading and processing data...",
                    "Data processed!",
                    [&] { all_data = processor_iter->second(); }
                );

                auto& [all_inputs, all_targets] = all_data; // Structured binding

                if (all_inputs.empty() || all_targets.empty() || all_inputs.size() != all_targets.size())
                {
                    throw std::runtime_error("Loaded dataset is empty or mismatched.");
                }

                num_attributes = all_inputs[0].size();
                num_targets    = all_targets[0].size();
                std::cout << "Dataset loaded: " << all_inputs.size() << " samples." << '\n';
                std::cout << "Input attributes per sample: " << num_attributes << '\n';
                std::cout << "Target outputs per sample: " << num_targets << '\n';

                // Split data into training and testing sets
                execute_timed(
                    "Splitting data...",
                    "Data split!",
                    [&]
                    {
                        auto [train_data, test_data] = split_data(all_inputs, all_targets, train_split_ratio);
                        train_inputs                 = std::move(train_data.first);
                        train_targets                = std::move(train_data.second);
                        test_inputs                  = std::move(test_data.first);
                        test_targets                 = std::move(test_data.second);
                    }
                );

                std::cout << "Training samples: " << train_inputs.size() << '\n';
                std::cout << "Testing samples: " << test_inputs.size() << '\n';

                if constexpr (DEV_MODE)
                {
                    if (!train_inputs.empty() && !train_targets.empty())
                    {
                        display_sample_data("Training Data Sample", train_inputs, train_targets, 5);
                    }
                    else
                    {
                        std::cout << "No training data generated to display." << '\n';
                    }
                }

                // Define Network Topology (example: input -> hidden1 -> hidden2 -> output)
                // Using smaller defaults for potentially smaller real datasets
                layer_sizes = {num_attributes, 16, 8, num_targets}; // Example: Adjust as needed

                // Get training parameters
                epochs        = get_input<std::size_t>("Enter number of training epochs", epochs);
                learning_rate = get_input<double>("Enter learning rate", learning_rate);
            }
            else if (dataset_choice == "Randomly Generated Data")
            {
                std::cout << "\nGenerating random data..." << '\n';
                num_attributes = get_input<std::size_t>("Enter number of input attributes", 10);
                num_targets = get_input<std::size_t>("Enter number of target classes (output neurons)", 4);
                const auto num_train_samples = get_input<std::size_t>("Enter number of training samples", 10000);
                const auto num_test_samples = get_input<std::size_t>("Enter number of test samples", 2000);
                epochs = get_input<std::size_t>("Enter number of training epochs", epochs);
                learning_rate = get_input<double>("Enter learning rate", learning_rate);

                // Define Network Topology (example: input -> hidden1 -> hidden2 -> output)
                layer_sizes = {num_attributes, 10, num_targets}; // Adjusted defaults

                // --- Data Generation ---
                std::random_device rd;
                std::mt19937       rng(rd()); // Mersenne Twister random number generator

                // Assuming generate_data_network fills the vectors passed by reference
                execute_timed(
                    "Generating training data...",
                    "Training data generated!",
                    generate_data_network,                           // Pass function directly
                    std::ref(train_inputs), std::ref(train_targets), // Pass vectors by reference
                    num_train_samples, num_attributes, num_targets, std::ref(rng)
                );

                execute_timed(
                    "Generating test data...",
                    "Test data generated!",
                    generate_data_network,
                    std::ref(test_inputs), std::ref(test_targets),
                    num_test_samples, num_attributes, num_targets, std::ref(rng)
                );

                if constexpr (DEV_MODE)
                {
                    if (!train_inputs.empty() && !train_targets.empty())
                    {
                        display_sample_data("Generated Training Data Sample", train_inputs, train_targets, 5);
                    }
                    else
                    {
                        std::cout << "No training data generated to display." << '\n';
                    }
                }
            }
            else // Invalid choice
            {
                throw std::runtime_error(std::format("Invalid dataset choice: '{}'", dataset_choice));
            }


            // --- Common Steps: Network Creation, Training, Evaluation ---
            if (train_inputs.empty() || test_inputs.empty())
            {
                throw std::runtime_error("No training or testing data available to proceed.");
            }

            // --- Activation Function Selection ---
            std::cout << "\n--- Activation Function Selection ---" << '\n';
            std::cout << "Choose activation function for HIDDEN layers:" << '\n';
            std::cout << "  1: ReLU" << '\n';
            std::cout << "  2: LeakyReLU (alpha=" << LRELU_ALPHA << ")" << '\n'; // Display alpha
            std::cout << "  3: Tanh" << '\n';
            std::cout << "  4: Sigmoid" << '\n';

            // Get hidden layer choice (Default to ReLU: 1)
            const int  hidden_choice            = get_input<int>("Enter choice for hidden layers", 1);
            const auto hidden_activation_choice = int_to_activation(hidden_choice);

            std::cout << "Choose activation function for OUTPUT layer:" << '\n';
            std::cout << "  (Usually Sigmoid for binary/multi-label, Tanh for regression [-1,1])" << '\n';
            std::cout << "  1: ReLU" << '\n';
            std::cout << "  2: LeakyReLU (alpha=" << LRELU_ALPHA << ")" << '\n';
            std::cout << "  3: Tanh" << '\n';
            std::cout << "  4: Sigmoid" << '\n';

            const int  output_choice            = get_input<int>("Enter choice for output layer", 4);
            const auto output_activation_choice = int_to_activation(output_choice);

            // --- Network Creation ---
            NeuralNetwork network = execute_timed(
                "Creating Neural Network...",
                "Neural Network created!",
                // Lambda ensures construction happens within the timed block
                [&] // Specify return type for clarity
                {
                    return NeuralNetwork(layer_sizes, hidden_activation_choice, output_activation_choice);
                    // Use topology defined above
                }
            );

            network.display_topology();

            // --- Network Training ---
            execute_timed(
                "Starting Network Training...",
                "Network Training finished!",
                // Use a lambda to call the member function
                // Pass gradient clipping threshold (e.g., 1.0) as the last arg if train supports it
                [&]
                {
                    // Example: Add clipping threshold if train method supports it
                    // network.train(train_inputs, train_targets, epochs, learning_rate, 1.0);
                    network.train(train_inputs, train_targets, epochs, learning_rate);
                    // Without explicit clipping arg shown
                }
            );

            // --- Network Evaluation ---
            if (!test_inputs.empty())
            {
                execute_timed(
                    "Starting Network Evaluation...",
                    "Network Evaluation finished!",
                    // Use a lambda to call the utility function
                    [&]
                    {
                        evaluate_network(network, test_inputs, test_targets);
                    }
                );
            }
            else
            {
                std::cout << "\nSkipping evaluation: No test data available." << '\n';
            }

            std::cout << '\n';

            if (auto exit_prompt = get_input<std::string>("Do you want to continue or quit? (y/n)", "n");
                exit_prompt == "n" || exit_prompt == "N")
            {
                running = false;
            }
        }
    } // End try block
    catch (const std::filesystem::filesystem_error& e)
    {
        std::cerr << "\n--- FILESYSTEM ERROR ---\n" << e.what() << "\nCode: " << e.code() << "\nPath1: " << e.path1() <<
                "\nPath2: " << e.path2() << "\n----------------------" << '\n';
        std::cin.get();
        return EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        std::cerr << "\n--- ERROR ---\n" << e.what() << "\n-------------" << '\n';
        std::cin.get();
        return EXIT_FAILURE; // Indicate failure
    }
    catch (...)
    {
        // Catch all other non-standard exceptions
        std::cerr << "\n--- UNKNOWN ERROR ---\nAn unexpected error occurred." << '\n';
        std::cin.get();
        return EXIT_FAILURE; // Indicate failure
    }

    // --- Successful Execution Exit ---
    std::cout << "\nExecution finished successfully." << '\n';
    std::cout << "Press ENTER to exit..." << '\n';
    // std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();

    return EXIT_SUCCESS; // Indicate success
}
