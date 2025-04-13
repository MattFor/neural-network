// Utilities.cpp
// Implementation file for utility functions.
// Created by MattFor on 13/04/2025. (Date corrected)

#include "../include/Utilities.h" // Include own header first
#include "../include/NeuralNetwork.h" // Need full definition for evaluate_network

// Other necessary standard library headers are included via Utilities.h now
double stringToDouble(const std::string& str) {
    try {
        return std::stod(str);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument("Invalid argument: Could not convert string to double: " + str);
    } catch (const std::out_of_range& e) {
        throw std::out_of_range("Out of range: Value too large or too small to convert to double: " + str);
    }
}

// Function to load and preprocess the zoo.csv file
void load_zoo_data(const std::string& filename, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets, std::size_t& num_attributes, std::size_t& num_targets) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::vector<int> unique_labels;

    // First pass to determine unique labels
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> row_data;
        while (std::getline(ss, token, ',')) {
            row_data.push_back(token);
        }
        if (row_data.size() > 1) {
            try {
                int label = std::stoi(row_data.back());
                if (std::find(unique_labels.begin(), unique_labels.end(), label) == unique_labels.end()) {
                    unique_labels.push_back(label);
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "Warning: Could not parse label in row: " << line << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Warning: Label out of range in row: " << line << std::endl;
            }
        }
    }
    std::sort(unique_labels.begin(), unique_labels.end());
    num_targets = unique_labels.size();

    // Second pass to load data and create one-hot targets
    file.clear();
    file.seekg(0, std::ios::beg);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> attribute_row;
        std::string label_str;
        std::vector<std::string> row_data;
        while (std::getline(ss, token, ',')) {
            row_data.push_back(token);
        }

        if (row_data.size() > 2) { // Ensure at least one attribute and a label
            // Extract attributes (skip the first element which is the animal name)
            for (size_t i = 1; i < row_data.size() - 1; ++i) {
                try {
                    attribute_row.push_back(stringToDouble(row_data[i]));
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Skipping attribute in row '" << line << "': " << e.what() << std::endl;
                    attribute_row.clear();
                    break;
                }
            }

            if (!attribute_row.empty()) {
                inputs.push_back(attribute_row);
                num_attributes = attribute_row.size(); // Set number of attributes based on the first valid row

                // Extract and one-hot encode the label
                label_str = row_data.back();
                try {
                    int label = std::stoi(label_str);
                    std::vector<double> target_row(num_targets, 0.0);
                    auto it = std::find(unique_labels.begin(), unique_labels.end(), label);
                    if (it != unique_labels.end()) {
                        size_t index = std::distance(unique_labels.begin(), it);
                        if (index < num_targets) {
                            target_row[index] = 1.0;
                        } else {
                            std::cerr << "Error: Calculated target index is out of bounds." << std::endl;
                        }
                    } else {
                        std::cerr << "Warning: Label '" << label << "' not found in unique labels." << std::endl;
                    }
                    targets.push_back(target_row);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Could not process label '" << label_str << "' in row '" << line << "': " << e.what() << std::endl;
                }
            }
        } else {
            std::cerr << "Warning: Skipping row with insufficient data: " << line << std::endl;
        }
    }

    file.close();

    if (inputs.empty()) {
        throw std::runtime_error("No data loaded from the CSV file.");
    }
}
// Display multi-label training data
void display_train_data(
    const std::vector<std::vector<double>>& train_inputs,
    const std::vector<std::vector<double>>& train_targets,
    const std::size_t num_attributes
)
{
    if (train_inputs.empty() || train_targets.empty()) {
        std::cout << "No training data to display." << std::endl;
        return;
    }
    if (train_inputs.size() != train_targets.size()){
         std::cerr << "Warning: Mismatch between number of input samples and target samples in display_train_data." << std::endl;
         // Decide whether to proceed or return
         // return;
    }

    // Determine approximate width needed for features display
    std::size_t feature_width = num_attributes * 6; // Estimate width based on number format below
    feature_width = std::max(feature_width, std::size_t{15}); // Minimum width

    std::cout << "\n--- Sample Training Data ---\n";
    // Print headers
    std::cout << std::left << std::setw(feature_width) << "Input Features" << " | " << "Target Vector (One-Hot)" << std::endl;
    std::cout << std::string(feature_width + 28, '-') << std::endl; // Adjust separator width

    // Print a few rows of training data (e.g., first 5 and last 5)
    const std::size_t display_count = 5;
    for (std::size_t i = 0; i < std::min(train_inputs.size(), display_count); ++i)
    {
        // Print feature vector
        std::cout << std::left << std::setw(feature_width);
        std::stringstream ss_features;
        for (std::size_t j = 0; j < train_inputs[i].size(); ++j)
        {
             ss_features << std::fixed << std::setprecision(2) << train_inputs[i][j] << (j < train_inputs[i].size() - 1 ? " " : "");
        }
        std::cout << ss_features.str();

        std::cout << " | ";

        // Print target vector
        if (i < train_targets.size()) { // Check bounds for targets as well
            for (const auto& val : train_targets[i])
            {
                std::cout << static_cast<int>(val) << " "; // Print targets as 0 or 1
            }
        }
        std::cout << std::endl;
    }

    if (train_inputs.size() > display_count * 2) {
         std::cout << std::left << std::setw(feature_width) << "..." << " | ..." << std::endl;
         for (std::size_t i = train_inputs.size() - display_count; i < train_inputs.size(); ++i) {
             // Print feature vector
             std::cout << std::left << std::setw(feature_width);
             std::stringstream ss_features;
             for (std::size_t j = 0; j < train_inputs[i].size(); ++j) {
                 ss_features << std::fixed << std::setprecision(2) << train_inputs[i][j] << (j < train_inputs[i].size() - 1 ? " " : "");
             }
             std::cout << ss_features.str();
             std::cout << " | ";
             // Print target vector
             if (i < train_targets.size()) {
                 for (const auto& val : train_targets[i]) {
                     std::cout << static_cast<int>(val) << " ";
                 }
             }
             std::cout << std::endl;
         }
    } else if (train_inputs.size() > display_count) {
        // Print remaining if fewer than 2*display_count but more than display_count
        for (std::size_t i = display_count; i < train_inputs.size(); ++i) {
             // Print feature vector
             std::cout << std::left << std::setw(feature_width);
             std::stringstream ss_features;
             for (std::size_t j = 0; j < train_inputs[i].size(); ++j) {
                 ss_features << std::fixed << std::setprecision(2) << train_inputs[i][j] << (j < train_inputs[i].size() - 1 ? " " : "");
             }
             std::cout << ss_features.str();
             std::cout << " | ";
             // Print target vector
             if (i < train_targets.size()) {
                 for (const auto& val : train_targets[i]) {
                     std::cout << static_cast<int>(val) << " ";
                 }
             }
             std::cout << std::endl;
        }
    }
     std::cout << std::string(feature_width + 28, '-') << std::endl;
     std::cout << std::endl;
}


// Generate synthetic data for network (finding max input index)
void generate_data_network(
    std::vector<std::vector<double>>& inputs,
    std::vector<std::vector<double>>& targets,
    const std::size_t                 num_samples,
    const std::size_t                 num_attributes,
    const std::size_t                 num_targets,
    std::mt19937&                     rng
)
{
    std::uniform_real_distribution<> input_dist(0.0, 1.0); // Inputs [0, 1]

    inputs.resize(num_samples, std::vector<double>(num_attributes));
    targets.resize(num_samples, std::vector<double>(num_targets, 0.0)); // Initialize targets with zeros

    for (std::size_t i = 0; i < num_samples; ++i)
    {
        // Generate random inputs
        for (std::size_t j = 0; j < num_attributes; ++j)
        {
            inputs[i][j] = input_dist(rng);
        }

        // Find the index of the largest input value
        // Use iterators for safety and generality
        auto max_it = std::max_element(inputs[i].begin(), inputs[i].end());

        if (max_it != inputs[i].end()) // Check if input vector was not empty
        {
            std::size_t target_class_index = std::distance(inputs[i].begin(), max_it);

            // Ensure the index is valid for the target vector size
            if (target_class_index < num_targets)
            {
                targets[i][target_class_index] = 1.0; // Set the one-hot target
            }
            else
            {
                 // This case should ideally not happen if num_targets >= num_attributes
                 // or if the task definition changes. Handle gracefully.
                 std::cerr << "Warning: Max input index (" << target_class_index
                           << ") exceeds number of target classes (" << num_targets
                           << ") for sample " << i << ". Setting target to class 0." << std::endl;
                 if (!targets[i].empty()) targets[i][0] = 1.0;
            }
        }
        else if (!targets[i].empty()) // Handle case of empty input vector
        {
             std::cerr << "Warning: Empty input vector generated for sample " << i << ". Setting target to class 0." << std::endl;
             targets[i][0] = 1.0; // Default target
        }
    }
    std::cout << "Generated " << num_samples << " samples for network training/testing." << std::endl;
}


// Evaluate the network using classification accuracy for one-hot encoded targets
void evaluate_network(
    NeuralNetwork& network, // Now const
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets
)
{
    if (inputs.empty()) {
        std::printf("Evaluation: No test data provided.\n");
        return;
    }
    if (inputs.size() != targets.size()) {
         throw std::runtime_error("Evaluation Error: Input and target counts mismatch.");
    }

    std::size_t correct_predictions = 0;
    double total_loss = 0.0;
    const std::size_t num_samples = inputs.size();

    // Create a non-const copy IF predict modifies internal state in a way
    // that interferes with subsequent predictions in the loop.
    // If predict is truly const w.r.t. weights/bias and only updates mutable caches,
    // the const reference is fine.
    // NeuralNetwork network_eval = network; // Example if copy needed

    for (std::size_t i = 0; i < num_samples; ++i) {
        // Use std::span for prediction
        std::span<const double> input_span(inputs[i]);
        // Call the const predict method
        std::vector<double> output_vec = network.predict(input_span);

        const auto& target_vec = targets[i];

         if (output_vec.size() != target_vec.size()) {
            throw std::runtime_error(std::format(
                "Evaluation Error: Output size ({}) mismatch with target size ({}) at index {}.",
                output_vec.size(), target_vec.size(), i));
        }
         if (target_vec.empty()){
             std::cerr << "Warning: Empty target vector encountered during evaluation at index " << i << std::endl;
             continue; // Skip this sample
         }

        // --- Accuracy Calculation (for Classification) ---
        // Find predicted class: index of the maximum output value
        auto predicted_it = std::max_element(output_vec.begin(), output_vec.end());
        std::size_t predicted_class = std::distance(output_vec.begin(), predicted_it);

        // Find actual class: index of the '1.0' in the one-hot target vector
        auto actual_it = std::find(target_vec.begin(), target_vec.end(), 1.0);
        if (actual_it != target_vec.end()) { // Ensure 1.0 was found
            std::size_t actual_class = std::distance(target_vec.begin(), actual_it);

            // Compare predicted class index with actual class index
            if (predicted_class == actual_class) {
                correct_predictions++;
            }
        } else {
             // This indicates the target vector wasn't properly one-hot encoded.
             std::cerr << "Warning: No '1.0' found in target vector during evaluation for sample " << i
                       << ". Check data generation/loading." << std::endl;
        }

         // --- Loss Calculation ---
         try {
              total_loss += NeuralNetwork::compute_loss(output_vec, target_vec);
         } catch (const std::exception& e) {
              std::cerr << "Warning: Error computing loss for sample " << i << ": " << e.what() << std::endl;
              // Optionally skip sample or assign max loss
         }
    } // End loop through samples

    // Calculate overall average loss and accuracy
    const double avg_loss = (num_samples == 0) ? 0.0 : total_loss / static_cast<double>(num_samples);
    const double accuracy = (num_samples == 0) ? 0.0 : static_cast<double>(correct_predictions) / static_cast<double>(num_samples);

    // Print the evaluation results
    std::printf("--- Evaluation Results ---\n");
    std::printf(" Average Loss: %.8f\n", avg_loss);
    std::printf(" Accuracy:     %.4f%% (%zu / %zu correct)\n",
                accuracy * 100.0, correct_predictions, num_samples);
    std::printf("--------------------------\n");
    std::fflush(stdout);
}