# MNIST 3-Layer Neural Network

This project is a C implementation of a 3-layer feed-forward neural network trained to recognize handwritten digits from the MNIST dataset. It uses backpropagation for training and supports Sigmoid or Tanh activation functions.

## Project Structure

- `code/`: Contains the source code.
  - `main.c`: Entry point and training/testing loop.
  - `nn.c/h`: Neural network implementation.
  - `mnist.c/h`: MNIST dataset loading and display utilities.
  - `Makefile`: Build instructions.
- `data/`: Directory for MNIST data files (must be downloaded separately if not present).
- `codigoGuia/`: Reference implementation.

## Requirements

- GCC compiler
- Make
- MNIST Dataset (idx-ubyte format) in the `data/` directory.

## Getting Started

### Build

Navigate to the `code/` directory and run:

```bash
make
```

This will create an executable in `code/bin/mnist-nn`.

### Run

Run the executable from the project root:

```bash
./code/bin/mnist-nn
```

## Implementation Details

- **Architecture**: 3 Layers (Input: 784 nodes, Hidden: 20 nodes, Output: 10 nodes).
- **Activation Function**: Sigmoid (default).
- **Learning Rate**: 0.2 (default).
- **Input**: Normalized 28x28 grayscale images from the MNIST dataset.

## Acknowledgments

Based on the work by Matt Lind.
- [Simple 3-Layer Neural Network for MNIST](http://mmlind.github.io/Simple_3_Layer_Neural_Network_for_MNIST_Handwriting_Recognition/)
- [Original Repository](http://github.com/mmlind/mnist-3lnn/)
