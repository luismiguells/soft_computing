/**
 * @file mnist.h
 * @brief Utilities for handling MNIST files
 */

#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>
#include <stdio.h>

#define MNIST_IMG_WIDTH 28
#define MNIST_IMG_HEIGHT 28

// Default file names
#define MNIST_TRAINING_SET_IMAGE_FILE_NAME "data/train-images-idx3-ubyte"
#define MNIST_TRAINING_SET_LABEL_FILE_NAME "data/train-labels-idx1-ubyte"
#define MNIST_TESTING_SET_IMAGE_FILE_NAME "data/t10k-images-idx3-ubyte"
#define MNIST_TESTING_SET_LABEL_FILE_NAME "data/t10k-labels-idx1-ubyte"

#define MNIST_MAX_TRAINING_IMAGES 60000
#define MNIST_MAX_TESTING_IMAGES 10000

typedef struct {
    uint8_t pixel[MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT];
} MNIST_Image;

typedef uint8_t MNIST_Label;

typedef struct {
    uint32_t magic_number;
    uint32_t max_images;
    uint32_t img_width;
    uint32_t img_height;
} MNIST_ImageFileHeader;

typedef struct {
    uint32_t magic_number;
    uint32_t max_images;
} MNIST_LabelFileHeader;

FILE *mnist_open_image_file(const char *file_name);
FILE *mnist_open_label_file(const char *file_name);

MNIST_Image mnist_get_image(FILE *image_file);
MNIST_Label mnist_get_label(FILE *label_file);

void mnist_display_image(const MNIST_Image *img, int lbl, int cls);
void mnist_clear_screen(void);

#endif // MNIST_H
