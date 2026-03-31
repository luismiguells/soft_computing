/**
 * @file main.c
 * @brief MNIST 3-layer neural network with dynamic path support
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "nn.h"
#include "mnist.h"

typedef struct {
    const char *train_images;
    const char *train_labels;
    const char *test_images;
    const char *test_labels;
} DatasetPaths;

static Vector *get_vector_from_image(const MNIST_Image *img) {
    Vector *v = malloc(sizeof(Vector) + (MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT * sizeof(double)));
    if (!v) return NULL;
    v->size = MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT;
    for (size_t i = 0; i < v->size; i++) {
        v->vals[i] = img->pixel[i] ? 1.0 : 0.0;
    }
    return v;
}

void train_network(Network *nn, const char *img_path, const char *lbl_path) {
    printf("Training network using:\n  Images: %s\n  Labels: %s\n", img_path, lbl_path);
    FILE *image_file = mnist_open_image_file(img_path);
    FILE *label_file = mnist_open_label_file(lbl_path);
    
    int err_count = 0;
    for (int i = 0; i < MNIST_MAX_TRAINING_IMAGES; i++) {
        MNIST_Image img = mnist_get_image(image_file);
        MNIST_Label lbl = mnist_get_label(label_file);
        
        Vector *inp = get_vector_from_image(&img);
        nn_feed_input(nn, inp);
        nn_feed_forward(nn);
        nn_back_propagate(nn, lbl);
        
        if (nn_get_classification(nn) != lbl) err_count++;
        free(inp);
        
        if (i % 5000 == 4999) {
            printf("Training progress: %5d/%d (Error rate: %.2f%%)\n", i + 1, MNIST_MAX_TRAINING_IMAGES, (double)err_count * 100.0 / (i + 1));
        }
    }
    
    fclose(image_file);
    fclose(label_file);
    printf("Training complete.\n\n");
}

void test_network(Network *nn, const char *img_path, const char *lbl_path) {
    printf("Testing network using:\n  Images: %s\n  Labels: %s\n", img_path, lbl_path);
    FILE *image_file = mnist_open_image_file(img_path);
    FILE *label_file = mnist_open_label_file(lbl_path);
    
    int err_count = 0;
    for (int i = 0; i < MNIST_MAX_TESTING_IMAGES; i++) {
        MNIST_Image img = mnist_get_image(image_file);
        MNIST_Label lbl = mnist_get_label(label_file);
        
        Vector *inp = get_vector_from_image(&img);
        nn_feed_input(nn, inp);
        nn_feed_forward(nn);
        
        int classification = nn_get_classification(nn);
        if (classification != lbl) err_count++;
        free(inp);
        
        if (i < 5) { // Show a few examples
            mnist_display_image(&img, lbl, classification);
            printf("\n");
        }
    }
    
    printf("Test Results: Correct=%d, Incorrect=%d, Accuracy=%.2f%%\n", 
           MNIST_MAX_TESTING_IMAGES - err_count, err_count, 
           (double)(MNIST_MAX_TESTING_IMAGES - err_count) * 100.0 / MNIST_MAX_TESTING_IMAGES);
    
    fclose(image_file);
    fclose(label_file);
}

void print_usage(const char *prog_name) {
    printf("Usage: %s [train_images train_labels test_images test_labels]\n", prog_name);
    printf("Defaults:\n");
    printf("  train_images: %s\n", MNIST_TRAINING_SET_IMAGE_FILE_NAME);
    printf("  train_labels: %s\n", MNIST_TRAINING_SET_LABEL_FILE_NAME);
    printf("  test_images:  %s\n", MNIST_TESTING_SET_IMAGE_FILE_NAME);
    printf("  test_labels:  %s\n", MNIST_TESTING_SET_LABEL_FILE_NAME);
}

int main(int argc, char *argv[]) {
    DatasetPaths paths = {
        MNIST_TRAINING_SET_IMAGE_FILE_NAME,
        MNIST_TRAINING_SET_LABEL_FILE_NAME,
        MNIST_TESTING_SET_IMAGE_FILE_NAME,
        MNIST_TESTING_SET_LABEL_FILE_NAME
    };

    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (argc == 5) {
        paths.train_images = argv[1];
        paths.train_labels = argv[2];
        paths.test_images = argv[3];
        paths.test_labels = argv[4];
    } else if (argc != 1) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    srand((unsigned int)time(NULL));
    mnist_clear_screen();
    printf("MNIST 3-Layer Neural Network\n\n");
    
    Network *nn = nn_create(MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT, 20, 10);
    if (!nn) {
        fprintf(stderr, "Failed to create network\n");
        return EXIT_FAILURE;
    }
    
    time_t start_time = time(NULL);
    train_network(nn, paths.train_images, paths.train_labels);
    test_network(nn, paths.test_images, paths.test_labels);
    time_t end_time = time(NULL);
    
    printf("\nTotal execution time: %.1f sec\n", difftime(end_time, start_time));
    
    nn_free(nn);
    return EXIT_SUCCESS;
}
