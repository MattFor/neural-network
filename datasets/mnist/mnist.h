//
// Created by MattFor on 14.04.2025.
//

#ifndef MNIST_H
#define MNIST_H

#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <utility>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "Utilities.h"

/**
 * @brief Loads the MNIST training and test CSVs, one per thread.
 *
 * Both files are parsed concurrently as work is dominated by string-to-double conversion rather than by disk I/O.
 *
 * @return { inputs, targets }: pixels scaled to [0, 1] and one-hot labels of length 10.
 */
inline std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> process_mnist()
{
    std::vector<std::vector<double>> input_data;
    std::vector<std::vector<double>> target_data;
    std::mutex                       data_mutex;

    // Function to process a single MNIST data file
    auto processFile = [&](const std::string& filename, std::vector<std::vector<double>>& local_input_data, std::vector<std::vector<double>>& local_target_data)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open MNIST file: " + filename + ". The CSVs are not bundled with this repository - see the README for where to download them.");
        }

        std::string line;
        std::getline(file, line); // Skip the header row

        while (std::getline(file, line))
        {
            std::stringstream        ss(line);
            std::string              segment;
            std::vector<std::string> features;

            while (std::getline(ss, segment, ','))
            {
                features.push_back(segment);
            }

            if (features.empty())
            {
                continue; // Skip empty lines
            }

            int label = 0;
            try
            {
                label = std::stoi(features[0]);
            }
            catch (const std::exception&)
            {
                std::cerr << "Warning: Unreadable label '" << features[0] << "' in file '" << filename << "'. Skipping row.\n";
                continue;
            }

            std::vector<double> input_instance;
            for (size_t i = 1; i < features.size(); i++)
            {
                try
                {
                    input_instance.push_back(std::stod(features[i]) / 255.0); // Normalize pixel values
                }
                catch (const std::invalid_argument&)
                {
                    std::cerr << "Warning: Invalid pixel value '" << features[i] << "' in file '" << filename << "' line: " << line << '\n';
                    input_instance.push_back(0.0);
                }
                catch (const std::out_of_range&)
                {
                    std::cerr << "Warning: Pixel value out of range '" << features[i] << "' in file '" << filename << "' line: " << line << '\n';
                    input_instance.push_back(0.0);
                }
            }

            local_input_data.push_back(input_instance);

            // Create a one-hot encoded target vector
            std::vector target_instance(10, 0.0);
            if (label >= 0 && label < 10)
            {
                target_instance[label] = 1.0;
            }
            else
            {
                std::cerr << "Warning: Invalid label '" << label << "' in file '" << filename << "' line: " << line << '\n';
            }

            local_target_data.push_back(target_instance);
        }

        file.close();
    };

    std::vector<std::vector<double>> train_input_data_local;
    std::vector<std::vector<double>> train_target_data_local;
    std::vector<std::vector<double>> test_input_data_local;
    std::vector<std::vector<double>> test_target_data_local;

    std::exception_ptr train_error;
    std::exception_ptr test_error;

    auto load_split = [&](const char* relative_path, std::vector<std::vector<double>>& local_inputs, std::vector<std::vector<double>>& local_targets, std::exception_ptr& error_slot)
    {
        try
        {
            processFile(resolve_project_path(relative_path).string(), local_inputs, local_targets);

            std::lock_guard lock(data_mutex);
            input_data.insert(input_data.end(), local_inputs.begin(), local_inputs.end());
            target_data.insert(target_data.end(), local_targets.begin(), local_targets.end());
        }
        catch (...)
        {
            error_slot = std::current_exception();
        }
    };

    std::thread train_thread(load_split, "datasets/mnist/mnist_train.csv", std::ref(train_input_data_local), std::ref(train_target_data_local), std::ref(train_error));

    std::thread test_thread(load_split, "datasets/mnist/mnist_test.csv", std::ref(test_input_data_local), std::ref(test_target_data_local), std::ref(test_error));

    train_thread.join();
    test_thread.join();

    if (train_error)
    {
        std::rethrow_exception(train_error);
    }

    if (test_error)
    {
        std::rethrow_exception(test_error);
    }

    return {
        std::move(input_data),
        std::move(target_data)
    };
}

#endif //MNIST_H
