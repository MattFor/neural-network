//
// Created by MattFor on 14.04.2025.
//

#ifndef CAR_H
#define CAR_H

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "Utilities.h"

namespace car_detail
{
    using Encoding = std::map<std::string, std::vector<double>>;

    // Six input attributes
    inline const std::vector<Encoding>& attribute_encodings()
    {
        static const std::vector<Encoding> encodings = {
            // buying
            {
                {
                    "vhigh",
                    {
                        1.0,
                        0.0,
                        0.0,
                        0.0
                    }
                },
                {
                    "high",
                    {
                        0.0,
                        1.0,
                        0.0,
                        0.0
                    }
                },
                {
                    "med",
                    {
                        0.0,
                        0.0,
                        1.0,
                        0.0
                    }
                },
                {
                    "low",
                    {
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    }
                }
            },
            // maint
            {
                {
                    "vhigh",
                    {
                        1.0,
                        0.0,
                        0.0,
                        0.0
                    }
                },
                {
                    "high",
                    {
                        0.0,
                        1.0,
                        0.0,
                        0.0
                    }
                },
                {
                    "med",
                    {
                        0.0,
                        0.0,
                        1.0,
                        0.0
                    }
                },
                {
                    "low",
                    {
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    }
                }
            },
            // doors
            {
                {
                    "2",
                    {
                        1.0,
                        0.0,
                        0.0,
                        0.0
                    }
                },
                {
                    "3",
                    {
                        0.0,
                        1.0,
                        0.0,
                        0.0
                    }
                },
                {
                    "4",
                    {
                        0.0,
                        0.0,
                        1.0,
                        0.0
                    }
                },
                {
                    "5more",
                    {
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    }
                }
            },
            // persons
            {
                {
                    "2",
                    {
                        1.0,
                        0.0,
                        0.0
                    }
                },
                {
                    "4",
                    {
                        0.0,
                        1.0,
                        0.0
                    }
                },
                {
                    "more",
                    {
                        0.0,
                        0.0,
                        1.0
                    }
                }
            },
            // lug_boot
            {
                {
                    "small",
                    {
                        1.0,
                        0.0,
                        0.0
                    }
                },
                {
                    "med",
                    {
                        0.0,
                        1.0,
                        0.0
                    }
                },
                {
                    "big",
                    {
                        0.0,
                        0.0,
                        1.0
                    }
                }
            },
            // safety
            {
                {
                    "low",
                    {
                        1.0,
                        0.0,
                        0.0
                    }
                },
                {
                    "med",
                    {
                        0.0,
                        1.0,
                        0.0
                    }
                },
                {
                    "high",
                    {
                        0.0,
                        0.0,
                        1.0
                    }
                }
            }
        };

        return encodings;
    }

    // Class label
    inline const Encoding& class_encoding()
    {
        static const Encoding classes = {
            {
                "unacc",
                {
                    1.0,
                    0.0,
                    0.0,
                    0.0
                }
            },
            {
                "acc",
                {
                    0.0,
                    1.0,
                    0.0,
                    0.0
                }
            },
            {
                "good",
                {
                    0.0,
                    0.0,
                    1.0,
                    0.0
                }
            },
            {
                "vgood",
                {
                    0.0,
                    0.0,
                    0.0,
                    1.0
                }
            }
        };

        return classes;
    }

    inline std::string trim(const std::string& text)
    {
        const std::size_t first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
            return {};
        }

        const std::size_t last = text.find_last_not_of(" \t\r\n");
        return text.substr(first, last - first + 1);
    }
}

/**
 * @brief Loads and one-hot encodes the Car Evaluation dataset.
 *
 * @return { inputs, targets } where each input has 21 values and each target has 4.
 *
 * @throws std::runtime_error if car.data cannot be opened, or contains no usable rows.
 */
inline std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> process_car()
{
    const std::filesystem::path path = resolve_project_path("datasets/car/car.data");

    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open car.data. Looked for: " + path.string());
    }

    std::string                      line;
    std::vector<std::vector<double>> input_data;
    std::vector<std::vector<double>> target_data;

    std::size_t line_number  = 0;
    std::size_t skipped_rows = 0;

    while (std::getline(file, line))
    {
        line_number++;

        std::stringstream        ss(line);
        std::string              segment;
        std::vector<std::string> features;

        while (std::getline(ss, segment, ','))
        {
            features.push_back(car_detail::trim(segment));
        }

        if (features.size() == 1 && features[0].empty())
        {
            continue; // Blank line
        }

        if (constexpr std::size_t expected_columns = 7; features.size() != expected_columns)
        {
            std::cerr << "Warning: car.data line " << line_number << " has " << features.size() << " fields, expected " << expected_columns << ". Skipping: " << line << '\n';
            skipped_rows++;
            continue;
        }

        std::vector<double> input_instance;
        bool                row_is_valid = true;

        for (std::size_t column = 0; column < car_detail::attribute_encodings().size(); column++)
        {
            const auto& encoding = car_detail::attribute_encodings()[column];
            const auto  match    = encoding.find(features[column]);

            if (match == encoding.end())
            {
                std::cerr << "Warning: car.data line " << line_number << " has unknown value '" << features[column] << "' in column " << column + 1 << ". Skipping row.\n";
                row_is_valid = false;
                break;
            }

            input_instance.insert(input_instance.end(), match->second.begin(), match->second.end());
        }

        if (!row_is_valid)
        {
            skipped_rows++;
            continue;
        }

        const auto class_match = car_detail::class_encoding().find(features.back());
        if (class_match == car_detail::class_encoding().end())
        {
            std::cerr << "Warning: car.data line " << line_number << " has unknown class '" << features.back() << "'. Skipping row.\n";
            skipped_rows++;
            continue;
        }

        input_data.push_back(std::move(input_instance));
        target_data.push_back(class_match->second);
    }

    if (input_data.empty())
    {
        throw std::runtime_error("car.data contained no usable rows: " + path.string());
    }

    if (skipped_rows > 0)
    {
        std::cerr << "Note: skipped " << skipped_rows << " malformed row(s) in car.data.\n";
    }

    return {
        std::move(input_data),
        std::move(target_data)
    };
}

#endif //CAR_H
