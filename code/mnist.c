/**
 * @file mnist.c
 * @brief Utilities for handling MNIST files
 */

#include "mnist.h"
#include <stdlib.h>
#include <string.h>

static uint32_t flip_bytes(uint32_t n) {
    uint32_t b0, b1, b2, b3;
    b0 = (n & 0x000000ff) << 24u;
    b1 = (n & 0x0000ff00) << 8u;
    b2 = (n & 0x00ff0000) >> 8u;
    b3 = (n & 0xff000000) >> 24u;
    return (b0 | b1 | b2 | b3);
}

FILE *mnist_open_image_file(const char *file_name) {
    FILE *f = fopen(file_name, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open MNIST image file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    
    MNIST_ImageFileHeader h;
    if (fread(&h, sizeof(MNIST_ImageFileHeader), 1, f) != 1) {
        fprintf(stderr, "Error reading image file header\n");
        exit(EXIT_FAILURE);
    }
    
    // Check magic number (should be 2051 for images)
    if (flip_bytes(h.magic_number) != 2051) {
        fprintf(stderr, "Warning: Unexpected magic number in image file: %d\n", flip_bytes(h.magic_number));
    }
    
    return f;
}

FILE *mnist_open_label_file(const char *file_name) {
    FILE *f = fopen(file_name, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open MNIST label file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    
    MNIST_LabelFileHeader h;
    if (fread(&h, sizeof(MNIST_LabelFileHeader), 1, f) != 1) {
        fprintf(stderr, "Error reading label file header\n");
        exit(EXIT_FAILURE);
    }
    
    // Check magic number (should be 2049 for labels)
    if (flip_bytes(h.magic_number) != 2049) {
        fprintf(stderr, "Warning: Unexpected magic number in label file: %d\n", flip_bytes(h.magic_number));
    }
    
    return f;
}

MNIST_Image mnist_get_image(FILE *image_file) {
    MNIST_Image img;
    if (fread(&img, sizeof(MNIST_Image), 1, image_file) != 1) {
        fprintf(stderr, "Error reading image\n");
        exit(EXIT_FAILURE);
    }
    return img;
}

MNIST_Label mnist_get_label(FILE *label_file) {
    MNIST_Label lbl;
    if (fread(&lbl, sizeof(MNIST_Label), 1, label_file) != 1) {
        fprintf(stderr, "Error reading label\n");
        exit(EXIT_FAILURE);
    }
    return lbl;
}

void mnist_clear_screen(void) {
    printf("\e[1;1H\e[2J");
}

void mnist_display_image(const MNIST_Image *img, int lbl, int cls) {
    for (int y = 0; y < MNIST_IMG_HEIGHT; y++) {
        for (int x = 0; x < MNIST_IMG_WIDTH; x++) {
            printf("%s", img->pixel[y * MNIST_IMG_WIDTH + x] > 127 ? "X" : ".");
        }
        printf("\n");
    }
    printf("Label: %d, Classification: %d\n", lbl, cls);
}
