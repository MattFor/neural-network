//
// Created by MattFor on 14.04.2025.
//

#ifndef MNIST_H
#define MNIST_H

inline std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> process_mnist()
{
    std::vector<std::vector<double>> input_data;
    std::vector<std::vector<double>> target_data;

    // Function to process a single MNIST data file
    auto processFile = [&](const std::string& filename)
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
            std::stringstream ss(line);
            std::string segment;
            std::vector<std::string> features;

            while (std::getline(ss, segment, ','))
            {
                features.push_back(segment);
            }

            if (features.empty())
            {
                continue; // Skip empty lines
            }

            int label = std::stoi(features[0]);
            std::vector<double> input_instance;
            for (size_t i = 1; i < features.size(); ++i)
            {
                try
                {
                    input_instance.push_back(std::stod(features[i]) / 255.0); // Normalize pixel values
                }
                catch (const std::invalid_argument& e)
                {
                    std::cerr << "Warning: Invalid pixel value '" << features[i] << "' in line: " << line << '\n';
                    input_instance.push_back(0.0); // Or handle the error as needed
                }
                catch (const std::out_of_range& e)
                {
                    std::cerr << "Warning: Pixel value out of range '" << features[i] << "' in line: " << line << '\n';
                    input_instance.push_back(0.0); // Or handle the error as needed
                }
            }

            input_data.push_back(input_instance);

            // Create one-hot encoded target vector
            std::vector<double> target_instance(10, 0.0);
            if (label >= 0 && label < 10)
            {
                target_instance[label] = 1.0;
            }
            else
            {
                std::cerr << "Warning: Invalid label '" << label << "' in line: " << line << '\n';
                // You might want to handle invalid labels differently, e.g., skip the instance
            }

            target_data.push_back(target_instance);
        }

        file.close();
    };

    try
    {
        processFile("../data/mnist/mnist_train.csv");
        processFile("../data/mnist/mnist_test.csv");
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << "Error processing MNIST data: " << error.what() << '\n';
        // Optionally re-throw the exception or return an empty pair
        throw;
    }

    return std::make_pair(input_data, target_data);
}

#endif //MNIST_H
