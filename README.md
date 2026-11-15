# Neural Network

[![Release](https://img.shields.io/github/v/release/MattFor/neural-network)](https://github.com/MattFor/neural-network/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A feedforward multithreaded neural network written from scratch in C++26, submitted as the final project for the algorithms and data structures class.

## Requirements

- GCC 13+, Clang 17+, or MSVC 19.36+
- CMake 3.20 or newer

## Usage

```sh
./build/NeuralNetwork
```

```
Available data:
- car.data
- mnist
- Randomly Generated Data
```

- **car.data** - the UCI _Car Evaluation_ set, bundled in `datasets/car/`. 1,728 samples, six categorical attributes
  one-hot encoded into 21 inputs, four output classes. Trains to ~100% test accuracy very quickly.
- **mnist** - handwritten digits. **The CSVs are not included**. Download
  `mnist_train.csv` and `mnist_test.csv` from
  [pjreddie.com/projects/mnist-in-csv](https://pjreddie.com/projects/mnist-in-csv/) and place them in `datasets/mnist/`.
- **Randomly Generated Data** - synthetic samples whose target is the index of the largest input feature. Useful for
  smoke-testing a topology without any data on disk.

Trained weights are written to `nn_data/` as binary files.

## Building

```sh
cmake -S . -B build
cmake --build build -j
```

### Tests

```sh
ctest --test-dir build --output-on-failure
```

## License

MIT, see [LICENSE](LICENSE). By [MattFor](https://github.com/MattFor).
