//
// Created by MattFor on 14.04.2025.
//

#ifndef CAR_H
#define CAR_H

inline std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> process_car()
{
    std::ifstream file("../datasets/car/car.data");
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file car.data");
    }

    std::string                      line;
    std::vector<std::vector<double>> input_data;
    std::vector<std::vector<double>> target_data;

    // Define mappings for categorical attributes
    std::map<std::string, std::vector<double>> buying_map =
    {
        {"vhigh", {1.0, 0.0, 0.0, 0.0}},
        {"high", {0.0, 1.0, 0.0, 0.0}},
        {"med", {0.0, 0.0, 1.0, 0.0}},
        {"low", {0.0, 0.0, 0.0, 1.0}}
    };

    std::map<std::string, std::vector<double>> maint_map =
    {
        {"vhigh", {1.0, 0.0, 0.0, 0.0}},
        {"high", {0.0, 1.0, 0.0, 0.0}},
        {"med", {0.0, 0.0, 1.0, 0.0}},
        {"low", {0.0, 0.0, 0.0, 1.0}}
    };

    std::map<std::string, std::vector<double>> doors_map =
    {
        {"2", {1.0, 0.0, 0.0, 0.0}},
        {"3", {0.0, 1.0, 0.0, 0.0}},
        {"4", {0.0, 0.0, 1.0, 0.0}},
        {"5more", {0.0, 0.0, 0.0, 1.0}}
    };

    std::map<std::string, std::vector<double>> persons_map =
    {
        {"2", {1.0, 0.0, 0.0}},
        {"4", {0.0, 1.0, 0.0}},
        {"more", {0.0, 0.0, 1.0}}
    };

    std::map<std::string, std::vector<double>> lug_boot_map =
    {
        {"small", {1.0, 0.0, 0.0}},
        {"med", {0.0, 1.0, 0.0}},
        {"big", {0.0, 0.0, 1.0}}
    };

    std::map<std::string, std::vector<double>> safety_map =
    {
        {"low", {1.0, 0.0, 0.0}},
        {"med", {0.0, 1.0, 0.0}},
        {"high", {0.0, 0.0, 1.0}}
    };

    std::map<std::string, std::vector<double>> class_map =
    {
        {"unacc", {1.0, 0.0, 0.0, 0.0}},
        {"acc", {0.0, 1.0, 0.0, 0.0}},
        {"good", {0.0, 0.0, 1.0, 0.0}},
        {"vgood", {0.0, 0.0, 0.0, 1.0}}
    };

    while (std::getline(file, line))
    {
        std::stringstream        ss(line);
        std::string              segment;
        std::vector<std::string> features;

        while (std::getline(ss, segment, ','))
        {
            features.push_back(segment);
        }

        if (features.size() != 7)
        {
            std::cerr << "Warning: Skipping invalid line with " << features.size() << " features: " << line << '\n';
            continue;
        }

        std::vector<double> input_instance;
        input_instance.insert(input_instance.end(), buying_map[features[0]].begin(), buying_map[features[0]].end());
        input_instance.insert(input_instance.end(), maint_map[features[1]].begin(), maint_map[features[1]].end());
        input_instance.insert(input_instance.end(), doors_map[features[2]].begin(), doors_map[features[2]].end());
        input_instance.insert(input_instance.end(), persons_map[features[3]].begin(), persons_map[features[3]].end());
        input_instance.insert(input_instance.end(), lug_boot_map[features[4]].begin(), lug_boot_map[features[4]].end());
        input_instance.insert(input_instance.end(), safety_map[features[5]].begin(), safety_map[features[5]].end());

        input_data.push_back(input_instance);
        target_data.push_back(class_map[features[6]]);
    }

    file.close();
    return std::make_pair(input_data, target_data);
}

#endif //CAR_H
