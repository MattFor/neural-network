#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

// Helper function to convert a string to double
// double stringToDouble(const std::string& str) {
//     try {
//         return std::stod(str);
//     } catch (const std::invalid_argument& e) {
//         throw std::invalid_argument("Invalid argument: Could not convert string to double: " + str);
//     } catch (const std::out_of_range& e) {
//         throw std::out_of_range("Out of range: Value too large or too small to convert to double: " + str);
//     }
// }
//
// // Function to load and preprocess the zoo.csv file
// void load_zoo_data(const std::string& filename, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets, std::size_t& num_attributes, std::size_t& num_targets) {
//     std::ifstream file(filename);
//     if (!file.is_open()) {
//         throw std::runtime_error("Could not open file: " + filename);
//     }
//
//     std::string line;
//     std::vector<int> unique_labels;
//
//     // First pass to determine unique labels
//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         std::string token;
//         std::vector<std::string> row_data;
//         while (std::getline(ss, token, ',')) {
//             row_data.push_back(token);
//         }
//         if (row_data.size() > 1) {
//             try {
//                 int label = std::stoi(row_data.back());
//                 if (std::find(unique_labels.begin(), unique_labels.end(), label) == unique_labels.end()) {
//                     unique_labels.push_back(label);
//                 }
//             } catch (const std::invalid_argument& e) {
//                 std::cerr << "Warning: Could not parse label in row: " << line << std::endl;
//             } catch (const std::out_of_range& e) {
//                 std::cerr << "Warning: Label out of range in row: " << line << std::endl;
//             }
//         }
//     }
//     std::sort(unique_labels.begin(), unique_labels.end());
//     num_targets = unique_labels.size();
//
//     // Second pass to load data and create one-hot targets
//     file.clear();
//     file.seekg(0, std::ios::beg);
//
//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         std::string token;
//         std::vector<double> attribute_row;
//         std::string label_str;
//         std::vector<std::string> row_data;
//         while (std::getline(ss, token, ',')) {
//             row_data.push_back(token);
//         }
//
//         if (row_data.size() > 2) { // Ensure at least one attribute and a label
//             // Extract attributes (skip the first element which is the animal name)
//             for (size_t i = 1; i < row_data.size() - 1; ++i) {
//                 try {
//                     attribute_row.push_back(stringToDouble(row_data[i]));
//                 } catch (const std::exception& e) {
//                     std::cerr << "Warning: Skipping attribute in row '" << line << "': " << e.what() << std::endl;
//                     attribute_row.clear();
//                     break;
//                 }
//             }
//
//             if (!attribute_row.empty()) {
//                 inputs.push_back(attribute_row);
//                 num_attributes = attribute_row.size(); // Set number of attributes based on the first valid row
//
//                 // Extract and one-hot encode the label
//                 label_str = row_data.back();
//                 try {
//                     int label = std::stoi(label_str);
//                     std::vector<double> target_row(num_targets, 0.0);
//                     auto it = std::find(unique_labels.begin(), unique_labels.end(), label);
//                     if (it != unique_labels.end()) {
//                         size_t index = std::distance(unique_labels.begin(), it);
//                         if (index < num_targets) {
//                             target_row[index] = 1.0;
//                         } else {
//                             std::cerr << "Error: Calculated target index is out of bounds." << std::endl;
//                         }
//                     } else {
//                         std::cerr << "Warning: Label '" << label << "' not found in unique labels." << std::endl;
//                     }
//                     targets.push_back(target_row);
//                 } catch (const std::exception& e) {
//                     std::cerr << "Warning: Could not process label '" << label_str << "' in row '" << line << "': " << e.what() << std::endl;
//                 }
//             }
//         } else {
//             std::cerr << "Warning: Skipping row with insufficient data: " << line << std::endl;
//         }
//     }
//
//     file.close();
//
//     if (inputs.empty()) {
//         throw std::runtime_error("No data loaded from the CSV file.");
//     }
// }