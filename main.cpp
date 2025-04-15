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
#include "datasets/DataTransformations.h"

// --- Main Program Loop ---
int main()
{
    try // Top-level error handling
    {
        std::cout << "--- Neural Network by MattFor ---\n" << '\n';

        // --- Dataset Processors Map ---
        const std::unordered_map<std::string, DataProcessingFunc> dataset_processors =
        {
            {"car.data", process_car},
            {"mnist", process_mnist}
            // Add more data here
        };

        // --- Persistent State ---
        std::vector<std::vector<double>> loaded_train_inputs,  loaded_test_inputs;
        std::vector<std::vector<double>> loaded_train_targets, loaded_test_targets;
        std::vector<std::size_t>         loaded_layer_sizes;
        auto                             loaded_hidden_activation = ActivationType::RELU;
        auto                             loaded_output_activation = ActivationType::SIGMOID;
        std::string                      current_dataset_name;
        bool                             dataset_loaded       = false;
        bool                             architecture_defined = false;

        bool running = true;

        while (running)
        {
            // --- Per-Iteration State ---
            bool weights_were_loaded = false; // Track if weights were loaded in this iteration
            std::size_t epochs = 100;
            double learning_rate = 0.01;
            constexpr double train_split_ratio = 0.8;
            std::size_t num_targets = 0;
            std::size_t num_attributes = 0;
            std::vector<std::size_t> layer_sizes;
            std::vector<std::vector<double>> train_inputs, test_inputs;
            std::vector<std::vector<double>> train_targets, test_targets;
            auto hidden_activation_choice = ActivationType::RELU;
            auto output_activation_choice = ActivationType::SIGMOID;

            // --- Dataset Selection ---
            std::cout << "Available data:" << '\n';
            for (const auto& name : std::ranges::views::keys(dataset_processors))
            {
                std::cout << " - " << name << '\n';
            }
            std::cout << " - Randomly Generated Data" << '\n';
            std::cout << '\n';

            // --- Data Loading and Processing ---
            if (auto dataset_choice = get_input<std::string>("Enter dataset name", "Randomly Generated Data");
                dataset_loaded && dataset_choice == current_dataset_name)
            {
                std::cout << "\nReusing previously loaded dataset: " << current_dataset_name << '\n';
                train_inputs             = loaded_train_inputs;
                train_targets            = loaded_train_targets;
                test_inputs              = loaded_test_inputs;
                test_targets             = loaded_test_targets;
                num_attributes           = loaded_train_inputs[0].size();
                num_targets              = loaded_train_targets[0].size();
                layer_sizes              = loaded_layer_sizes;
                hidden_activation_choice = loaded_hidden_activation;
                output_activation_choice = loaded_output_activation;
            }
            else
            {
                // --- Processing based on choice ---
                if (const auto processor_iter = dataset_processors.find(dataset_choice);
                    processor_iter != dataset_processors.end()) // Existing dataset chosen
                {
                    std::cout << "\nProcessing dataset: " << dataset_choice << "..." << '\n';
                    if (!std::filesystem::exists(dataset_choice))
                    {
                        std::cerr << "Warning: Data file '" << dataset_choice
                                << "' not found in the execution directory. Processing might fail." << '\n';
                    }

                    DataPair all_data;
                    execute_timed(
                        "Loading and processing data...",
                        "Data processed!",
                        [&] { all_data = processor_iter->second(); }
                    );

                    auto& [all_inputs, all_targets] = all_data;
                    if (all_inputs.empty() || all_targets.empty() || all_inputs.size() != all_targets.size())
                    {
                        throw std::runtime_error("Loaded dataset is empty or mismatched.");
                    }

                    num_attributes = all_inputs[0].size();
                    num_targets    = all_targets[0].size();
                    std::cout << "Dataset loaded: " << all_inputs.size() << " samples." << '\n';
                    std::cout << "Input attributes per sample: " << num_attributes << '\n';
                    std::cout << "Target outputs per sample: " << num_targets << '\n';

                    execute_timed(
                        "Splitting data...",
                        "Data split!",
                        [&]
                        {
                            auto [train_data_pair, test_data_pair] = split_data(
                                all_inputs, all_targets, train_split_ratio);
                            train_inputs  = std::move(train_data_pair.first);
                            train_targets = std::move(train_data_pair.second);
                            test_inputs   = std::move(test_data_pair.first);
                            test_targets  = std::move(test_data_pair.second);
                        }
                    );

                    std::cout << "Training samples: " << train_inputs.size() << '\n';
                    std::cout << "Testing samples: " << test_inputs.size() << '\n';

                    if constexpr (DEV_MODE)
                    {
                        if (!train_inputs.empty())
                        {
                            display_sample_data("Training Data Sample", train_inputs, train_targets, 5);
                        }
                        else
                        {
                            std::cout << "No training data generated to display." << '\n';
                        }
                    }

                    // Store loaded dataset
                    loaded_train_inputs  = train_inputs;
                    loaded_train_targets = train_targets;
                    loaded_test_inputs   = test_inputs;
                    loaded_test_targets  = test_targets;
                    current_dataset_name = dataset_choice;
                    dataset_loaded       = true;
                    architecture_defined = false; // Reset architecture as dataset might have changed
                }
                else if (dataset_choice == "Randomly Generated Data")
                {
                    std::cout << "\nGenerating random data..." << '\n';
                    num_attributes = get_input<std::size_t>("Enter number of input attributes", 10);
                    num_targets = get_input<std::size_t>("Enter number of target classes (output neurons)", 4);
                    const auto num_train_samples = get_input<std::size_t>("Enter number of training samples", 10000);
                    const auto num_test_samples = get_input<std::size_t>("Enter number of test samples", 2000);

                    // Define Network Topology (for random data, we define it before generating)
                    std::size_t hidden1_size = std::max(static_cast<size_t>(10), (num_attributes + num_targets) / 2);
                    loaded_layer_sizes       = {num_attributes, hidden1_size, num_targets};
                    architecture_defined     = true;
                    layer_sizes              = loaded_layer_sizes; // Set for this iteration as well

                    std::random_device rd;
                    std::mt19937       rng(rd());

                    execute_timed("Generating training data...", "Training data generated!",
                                  generate_data_network, std::ref(train_inputs), std::ref(train_targets),
                                  num_train_samples, num_attributes, num_targets, std::ref(rng));

                    execute_timed("Generating test data...", "Test data generated!",
                                  generate_data_network, std::ref(test_inputs), std::ref(test_targets),
                                  num_test_samples, num_attributes, num_targets, std::ref(rng));

                    if constexpr (DEV_MODE)
                    {
                        if (!train_inputs.empty())
                        {
                            display_sample_data("Generated Training Data Sample", train_inputs, train_targets, 5);
                        }
                        else
                        {
                            std::cout << "No training data generated to display." << '\n';
                        }
                    }

                    // Store generated dataset
                    loaded_train_inputs  = train_inputs;
                    loaded_train_targets = train_targets;
                    loaded_test_inputs   = test_inputs;
                    loaded_test_targets  = test_targets;
                    current_dataset_name = dataset_choice;
                    dataset_loaded       = true;
                }
                else // Invalid choice
                {
                    throw std::runtime_error("Invalid dataset choice: '" + dataset_choice + "'");
                }
            }

            // --- Network Architecture Selection ---
            if (!architecture_defined || layer_sizes.empty())
            {
                NetworkComplexity complexity;

                auto complexity_choice = get_input<std::string>(
                    "Do you want to set the network complexity yourself (y), use a predefined level (p), or automatically determine it (a)?",
                    "a");

                if (train_inputs[0].size() < 10)
                {
                    complexity = NetworkComplexity::SIMPLE;
                }
                else if (train_inputs[0].size() < 150)
                {
                    complexity = NetworkComplexity::MEDIUM;
                }
                else
                {
                    complexity = NetworkComplexity::COMPLEX;
                }

                // Generate the layer sizes
                execute_timed(
                    "Generating network topology...",
                    "Topology generated!",
                    [&]
                    {
                        if (complexity_choice == "y")
                        {
                            std::cout << "Enter the layer sizes separated by spaces (f.e " << num_attributes <<
                                    " 128 " << num_targets << ") [FIRST AND LAST LAYER MUST ADHERE TO THE EXAMPLE]:\n";

                            std::string layer_sizes_str;
                            std::getline(std::cin, layer_sizes_str);
                            std::stringstream ss(layer_sizes_str);
                            std::size_t       layer_size;
                            layer_sizes.clear();

                            while (ss >> layer_size)
                            {
                                layer_sizes.push_back(layer_size);
                            }

                            if (layer_sizes.empty())
                            {
                                std::cerr << "Error: No layer sizes provided. Falling back to automatic complexity.\n";
                                complexity_choice = "a";
                            }
                            else if (layer_sizes.front() != num_attributes)
                            {
                                std::cerr << "Error: The first layer size must be equal to the number of attributes ("
                                        << num_attributes << "). Falling back to automatic complexity.\n";
                                complexity_choice = "a";
                            }
                            else if (layer_sizes.back() != num_targets)
                            {
                                std::cerr << "Error: The last layer size must be equal to the number of targets (" <<
                                        num_targets << "). Falling back to automatic complexity.\n";
                                complexity_choice = "a";
                            }
                            else
                            {
                                std::cout << "User-defined layer sizes accepted.\n";
                                loaded_layer_sizes   = layer_sizes;
                                architecture_defined = true;
                                return;
                            }
                        }

                        if (complexity_choice == "p" || complexity_choice == "a")
                        {
                            int complexity_level_input;

                            if (complexity_choice == "p")
                            {
                                complexity_level_input = get_input<int>(
                                    "Enter the complexity level (0: Simple, 1: Medium, 2: Complex):", 1);
                            }
                            else // complexity_choice == "a"
                            {
                                if (train_inputs[0].size() < 10)
                                {
                                    complexity_level_input = 0;
                                }
                                else if (train_inputs[0].size() < 150)
                                {
                                    complexity_level_input = 1;
                                }
                                else
                                {
                                    complexity_level_input = 2;
                                }

                                std::cout << "Automatically determined complexity level: " << complexity_level_input <<
                                        std::endl;
                            }

                            switch (complexity_level_input)
                            {
                                case 0:
                                {
                                    complexity = NetworkComplexity::SIMPLE;
                                }
                                break;

                                case 1:
                                {
                                    complexity = NetworkComplexity::MEDIUM;
                                }
                                break;

                                case 2:
                                {
                                    complexity = NetworkComplexity::COMPLEX;
                                }
                                break;

                                default:
                                {
                                    std::cerr << "Warning: Invalid complexity level. Using Medium.\n";
                                    complexity = NetworkComplexity::MEDIUM;
                                }
                                break;
                            }

                            loaded_layer_sizes = NeuralNetwork::generate_layer_topology(
                                num_attributes,
                                num_targets,
                                train_inputs.size(),
                                complexity
                            );

                            layer_sizes          = loaded_layer_sizes;
                            architecture_defined = true;
                        }
                        else
                        {
                            std::cerr <<
                                    "Warning: Invalid choice for complexity. Falling back to automatic complexity.\n";

                            if (train_inputs[0].size() < 10)
                            {
                                complexity = NetworkComplexity::SIMPLE;
                            }
                            else if (train_inputs[0].size() < 150)
                            {
                                complexity = NetworkComplexity::MEDIUM;
                            }
                            else
                            {
                                complexity = NetworkComplexity::COMPLEX;
                            }

                            loaded_layer_sizes = NeuralNetwork::generate_layer_topology(
                                num_attributes,
                                num_targets,
                                train_inputs.size(),
                                complexity
                            );
                            layer_sizes          = loaded_layer_sizes;
                            architecture_defined = true;
                        }
                    }
                );

                std::cout << "Generated layer sizes: ";
                for (size_t i = 0; i < layer_sizes.size(); i++)
                {
                    std::cout << (i > 0 ? " -> " : "") << layer_sizes[i];
                }
                std::cout << std::endl;

                // --- Activation Function Selection ---
                std::cout << "\n--- Activation Function Selection ---" << '\n';
                std::cout << "Choose activation function for HIDDEN layers:" << '\n';
                std::cout << "  1: ReLU\n  2: LeakyReLU (alpha=" << LRELU_ALPHA << ")\n  3: Tanh\n  4: Sigmoid\n";
                const int hidden_choice  = get_input<int>("Enter choice for hidden layers", 1);
                loaded_hidden_activation = int_to_activation(hidden_choice);
                hidden_activation_choice = loaded_hidden_activation;

                std::cout << "Choose activation function for OUTPUT layer:" << '\n';
                std::cout << "  (Usually Sigmoid for binary/multi-label, Tanh for regression [-1,1])" << '\n';
                std::cout << "  1: ReLU\n  2: LeakyReLU (alpha=" << LRELU_ALPHA << ")\n  3: Tanh\n  4: Sigmoid\n";
                const int output_choice  = get_input<int>("Enter choice for output layer", 4);
                loaded_output_activation = int_to_activation(output_choice);
                output_activation_choice = loaded_output_activation;
            }
            else
            {
                std::cout << "\nReusing previously defined network architecture." << std::endl;
                layer_sizes              = loaded_layer_sizes;
                hidden_activation_choice = loaded_hidden_activation;
                output_activation_choice = loaded_output_activation;
            }

            // --- Common Steps: Network Creation, Activation, Load/Train, Evaluate, Save ---
            if (train_inputs.empty() || test_inputs.empty())
            {
                throw std::runtime_error("No training or testing data available to proceed.");
            }

            // --- Network Creation ---
            // Create the network structure based on selected data and activations
            NeuralNetwork network = execute_timed(
                "Creating Neural Network Structure...",
                "Neural Network structure created!",
                [&]
                {
                    return NeuralNetwork(layer_sizes, hidden_activation_choice, output_activation_choice);
                }
            );

            network.display_topology();

            // --- Load Weights or Train ---
            std::cout << "\n--- Action ---" << '\n';
            auto action_choice = get_input<std::string>(
                "Do you want to [T]rain the network, [L]oad weights, or [C]ontinue training from loaded weights? (T/L/C)",
                "T");

            if (action_choice == "L" || action_choice == "l" || action_choice == "C" || action_choice == "c")
            {
                auto load_filename = get_input<std::string>("Enter filename to load weights from",
                                                            "network_weights.bin");

                if (std::filesystem::exists("../nn_data/" + load_filename))
                {
                    bool load_success = false;
                    execute_timed(
                        std::format("Attempting to load weights from '{}'...", load_filename),
                        "Weight loading attempt finished.", // Message neutral as success checked below
                        [&]
                        {
                            load_success = network.load_weights(load_filename);
                        }
                    );

                    if (load_success)
                    {
                        std::cout << "Weights loaded successfully. Network state restored." << '\n';
                        weights_were_loaded = true; // Set flag

                        if (auto display_topology_prompt = get_input<std::string>(
                            "Do you want to display loaded topology? (y/n)",
                            "n"); display_topology_prompt == "y" || display_topology_prompt == "Y")
                        {
                            network.display_topology();
                        }

                        if (action_choice == "C" || action_choice == "c")
                        {
                            std::cout << "\n--- Continue Network Training ---" << '\n';
                            epochs = get_input<std::size_t>("Enter number of additional training epochs", epochs);
                            learning_rate = get_input<double>("Enter learning rate for continued training",
                                                              learning_rate);

                            if (train_inputs.empty() || train_targets.empty())
                            {
                                std::cerr << "Cannot continue training: No training data available." << '\n';
                            }
                            else
                            {
                                execute_timed(
                                    "",
                                    "Network Training finished!",
                                    [&]
                                    {
                                        network.train(train_inputs, train_targets, epochs, learning_rate);
                                    }
                                );

                                // --- Ask whether to save weights after continued training ---
                                std::cout << "\n--- Save Weights ---" << '\n';
                                if (get_input<std::string>("Do you want to save the trained weights? (y/n)", "n") ==
                                    "y")
                                {
                                    std::stringstream ss;
                                    ss << "network_weights";
                                    for (size_t size : network.get_layer_sizes())
                                    {
                                        ss << "_" << size;
                                    }
                                    ss << ".bin";
                                    std::string suggested_filename = ss.str();
                                    auto save_filename = get_input<std::string>("Enter filename to save weights to",
                                        suggested_filename);
                                    execute_timed(
                                        std::format("Saving weights to '{}'...", save_filename),
                                        "Weights save operation complete.",
                                        [&]
                                        {
                                            network.save_weights(save_filename);
                                        }
                                    );
                                }
                            }
                        }
                    }
                    else
                    {
                        std::cerr << "Failed to load weights. Check file and network configuration." << '\n';
                        std::cout << "Proceeding with training a new network." << '\n';
                        // weights_were_loaded remains false
                    }
                }
                else
                {
                    std::cerr << "Weight file '" << load_filename << "' not found. Proceeding with training." << '\n';
                    // weights_were_loaded remains false
                }
            }

            // --- Network Training (Conditional - if not loaded and not continuing) ---
            if (!weights_were_loaded && (action_choice == "T" || action_choice == "t"))
            {
                std::cout << "\n--- Starting Network Training ---" << '\n';
                if (train_inputs.empty() || train_targets.empty())
                {
                    std::cerr << "Cannot train: No training data available." << '\n';
                }
                else
                {
                    // --- Get training parameters ---
                    epochs        = get_input<std::size_t>("Enter number of training epochs", epochs);
                    learning_rate = get_input<double>("Enter learning rate", learning_rate);

                    execute_timed(
                        "",
                        "Network Training finished!",
                        [&]
                        {
                            network.train(train_inputs, train_targets, epochs, learning_rate);
                        }
                    );

                    // --- Ask whether to save weights ---
                    std::cout << "\n--- Save Weights ---" << '\n';
                    if (get_input<std::string>("Do you want to save the trained weights? (y/n)", "n") == "y")
                    {
                        std::stringstream ss;
                        ss << "network_weights";

                        // Include dataset name in the filename
                        if (!current_dataset_name.empty())
                        {
                            // Sanitize the dataset name to be filename-friendly
                            std::string sanitized_dataset_name = current_dataset_name;
                            std::replace(sanitized_dataset_name.begin(), sanitized_dataset_name.end(), '.', '_');
                            std::replace(sanitized_dataset_name.begin(), sanitized_dataset_name.end(), ' ', '_');
                            ss << "_" << sanitized_dataset_name;
                        }
                        else
                        {
                            ss << "_unknown_dataset"; // Fallback if dataset name is not available
                        }

                        for (size_t size : network.get_layer_sizes())
                        {
                            ss << "_" << size;
                        }
                        ss << ".bin";
                        std::string suggested_filename = ss.str();

                        auto save_filename = get_input<std::string>("Enter filename to save weights to",
                                                                    suggested_filename);

                        execute_timed(
                            std::format("Saving weights to '{}'...", "nn_data/" + save_filename),
                            "Weights save operation complete.",
                            [&]
                            {
                                network.save_weights(save_filename);
                            }
                        );
                    }
                }
            }
            else if (weights_were_loaded && (action_choice != "C" && action_choice != "c"))
            {
                std::cout << "\nSkipping training because weights were loaded." << '\n';
            }

            // --- Network Evaluation (Always perform if test data exists) ---
            std::cout << "\n--- Network Evaluation ---" << '\n';
            if (!test_inputs.empty())
            {
                execute_timed(
                    "Starting Network Evaluation...",
                    "Network Evaluation finished!",
                    [&]
                    {
                        evaluate_network(network, test_inputs, test_targets);
                    }
                );
            }
            else
            {
                std::cout << "Skipping evaluation: No test data available." << '\n';
            }

            // --- Loop Continuation ---
            std::cout << '\n';
            if (auto exit_prompt = get_input<std::string>("Do you want to run another cycle? (y/n)", "n");
                exit_prompt == "n" || exit_prompt == "N")
            {
                running = false;
            }

            std::cout << "\n========================================\n" << '\n'; // Separator for next cycle
        }                                                                        // End while(running) loop
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        std::cerr << "\n--- FILESYSTEM ERROR ---\n" << e.what() << "\nCode: " << e.code() << "\nPath1: " << e.path1() <<
                "\nPath2: " << e.path2() << "\n----------------------" << '\n';
        std::cin.get();
        return EXIT_FAILURE;
    } catch (const std::exception& e)
    {
        std::cerr << "\n--- ERROR ---\n" << e.what() << "\n-------------" << '\n';
        std::cin.get();
        return EXIT_FAILURE;
    } catch (...)
    {
        std::cerr << "\n--- UNKNOWN ERROR ---\nAn unexpected error occurred." << '\n';
        std::cin.get();
        return EXIT_FAILURE;
    }

    // --- Successful Execution Exit ---
    std::cout << "\nExecution finished successfully." << '\n';
    std::cout << "Press ENTER to exit..." << '\n';
    // This is not useful here anymore.
    // std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();

    return EXIT_SUCCESS;
}
