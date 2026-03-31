# MNIST 3-Layer Neural Network (Soft Computing Project)

A refactored, high-performance C implementation of a 3-layer feed-forward neural network designed to identify handwritten digits from the MNIST dataset.

This project is a clean, modernized version of the original "3lnn" and "RNA" implementations, featuring improved error handling, consistent naming, and professional C project structure.

## Key Features

- **Refactored Architecture**: Modular design separating neural network logic (`nn.c/h`) from dataset handling (`mnist.c/h`).
- **Standardized Implementation**: Follows C99 standards with robust pointer arithmetic and memory management.
- **Dynamic Configuration**: Supports Sigmoid and Tanh activation functions with adjustable learning rates.
- **Detailed Progress Tracking**: Real-time training accuracy and error rate reporting.
- **Visual Verification**: Renders digits directly in the terminal for visual testing.

## Project Structure

- `code/`: **(Refactored)** The main project source code.
  - `nn.c/h`: Core neural network engine (Backpropagation, Forward Feed).
  - `mnist.c/h`: Dataset loader and terminal-based image rendering.
  - `main.c`: Training and testing orchestration.
  - `Makefile`: Optimized build system.
- `data/`: Required MNIST dataset files (`train-images`, `train-labels`, etc.).
- `codigoGuia/`: Reference implementation and original source material for comparison.

## 🛠 Prerequisites

Ensure you have the following installed:
- **GCC** (GNU Compiler Collection)
- **Make**

## Dataset Download

The MNIST dataset consists of four files in the `idx-ubyte` format. You must download and extract them into the `data/` directory.

### Official Source (LeCun's Website):
- [train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz) (9.9 MB)
- [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz) (29 KB)
- [t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz) (1.6 MB)
- [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz) (5 KB)

### Reliable Mirror (AWS S3):
If the official site is unreachable, use these links:
- [Training Images](https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz)
- [Training Labels](https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz)
- [Test Images](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz)
- [Test Labels](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz)

### Setup Instructions:
1.  Download the four `.gz` files.
2.  Decompress them using `gzip -d` or any extraction tool.
3.  Place the extracted files into the `data/` folder. Ensure the names match exactly (remove the `.gz` extension):
    - `data/train-images-idx3-ubyte`
    - `data/train-labels-idx1-ubyte`
    - `data/t10k-images-idx3-ubyte`
    - `data/t10k-labels-idx1-ubyte`


## Build & Run

### Build the project
Navigate to the `code/` directory and compile:
```bash
cd code
make
```

### 2. Run the application
Execute the binary from the project root. It will use default paths if no arguments are provided:
```bash
./code/bin/mnist-nn
```

**Using Dynamic Paths:**
You can specify custom dataset paths by passing them as arguments:
```bash
./code/bin/mnist-nn <train_images> <train_labels> <test_images> <test_labels>
```
Example:
```bash
./code/bin/mnist-nn data/train-images-idx3-ubyte data/train-labels-idx1-ubyte data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte
```
Use `./code/bin/mnist-nn --help` for more information.


## Technical Details

- **Input Layer**: 784 nodes (28x28 pixels).
- **Hidden Layer**: 20 nodes (configurable).
- **Output Layer**: 10 nodes (representing digits 0-9).
- **Backpropagation**: Adjusts weights based on the difference between target labels and network predictions.
- **Normalization**: Pixel values (0-255) are converted to a binary format (0 or 1) for the network input.

## Acknowledgments

This implementation is based on the foundational work by **Matt Lind**.
- [Original Article: Simple 3-Layer Neural Network](http://mmlind.github.io/Simple_3_Layer_Neural_Network_for_MNIST_Handwriting_Recognition/)
- [Original Github Repository](http://github.com/mmlind/mnist-3lnn/)

---
*Developed as part of the Soft Computing course.*
