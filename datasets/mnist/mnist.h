//
// Created by MattFor on 14.04.2025.
//

#ifndef MNIST_H
#define MNIST_H

#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <fstream>
#include <sstream>
#include <iostream>

/**
 * @brief I know, multithreaded file loading, but trust me this will be faster than doing it sequentially
 */
inline std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> process_mnist()
{
    std::vector<std::vector<double>> input_data;
    std::vector<std::vector<double>> target_data;
    std::mutex                       data_mutex;

    // Function to process a single MNIST data file
    auto processFile = [&](
        const std::string&                filename, std::vector<std::vector<double>>& local_input_data,
        std::vector<std::vector<double>>& local_target_data
    )
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file: " + filename);
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

            int                 label = std::stoi(features[0]);
            std::vector<double> input_instance;
            for (size_t i = 1; i < features.size(); i++)
            {
                try
                {
                    input_instance.push_back(std::stod(features[i]) / 255.0); // Normalize pixel values
                }
                catch (const std::invalid_argument& e)
                {
                    std::cerr << "Warning: Invalid pixel value '" << features[i] << "' in file '" << filename <<
                            "' line: " << line << '\n';
                    input_instance.push_back(0.0);
                }
                catch (const std::out_of_range& e)
                {
                    std::cerr << "Warning: Pixel value out of range '" << features[i] << "' in file '" << filename <<
                            "' line: " << line << '\n';
                    input_instance.push_back(0.0);
                }
            }

            local_input_data.push_back(input_instance);

            // Create one-hot encoded target vector
            std::vector target_instance(10, 0.0);
            if (label >= 0 && label < 10)
            {
                target_instance[label] = 1.0;
            }
            else
            {
                std::cerr << "Warning: Invalid label '" << label << "' in file '" << filename << "' line: " << line <<
                        '\n';
            }

            local_target_data.push_back(target_instance);
        }

        file.close();
    };

    std::vector<std::vector<double>> train_input_data_local;
    std::vector<std::vector<double>> train_target_data_local;
    std::vector<std::vector<double>> test_input_data_local;
    std::vector<std::vector<double>> test_target_data_local;

    std::thread train_thread([&]
    {
        try
        {
            processFile("../datasets/mnist/mnist_train.csv", train_input_data_local, train_target_data_local);
            std::lock_guard lock(data_mutex);
            input_data.insert(input_data.end(), train_input_data_local.begin(), train_input_data_local.end());
            target_data.insert(target_data.end(), train_target_data_local.begin(), train_target_data_local.end());
        }
        catch (const std::runtime_error& error)
        {
            std::cerr << "Error processing MNIST training data in thread: " << error.what() << '\n';
        }
    });

    std::thread test_thread([&]
    {
        try
        {
            processFile("../datasets/mnist/mnist_test.csv", test_input_data_local, test_target_data_local);
            std::lock_guard lock(data_mutex);
            input_data.insert(input_data.end(), test_input_data_local.begin(), test_input_data_local.end());
            target_data.insert(target_data.end(), test_target_data_local.begin(), test_target_data_local.end());
        }
        catch (const std::runtime_error& error)
        {
            std::cerr << "Error processing MNIST testing data in thread: " << error.what() << '\n';
        }
    });

    train_thread.join();
    test_thread.join();

    return std::make_pair(input_data, target_data);
}

#endif //MNIST_H
