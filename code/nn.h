/**
 * @file nn.h
 * @brief 3-layer neural network implementation (Input, Hidden, Output)
 * 
 * Supports Sigmoid and Tanh activation functions.
 */

#ifndef NN_H
#define NN_H

#include <stddef.h>

typedef enum {
    LAYER_INPUT,
    LAYER_HIDDEN,
    LAYER_OUTPUT
} LayerType;

typedef enum {
    ACT_SIGMOID,
    ACT_TANH
} ActivationType;

/**
 * @brief Dynamic data structure containing a defined number of values.
 */
typedef struct {
    size_t size;
    double vals[];
} Vector;

/**
 * @brief Dynamic data structure modeling a neuron with weights and bias.
 */
typedef struct {
    double bias;
    double output;
    size_t wcount;
    double weights[];
} Node;

/**
 * @brief Dynamic data structure holding a definable number of nodes that form a layer.
 */
typedef struct {
    size_t ncount;
    Node nodes[];
} Layer;

/**
 * @brief Data structure holding the whole network.
 */
typedef struct {
    size_t inp_node_size;
    size_t inp_layer_size;
    size_t hid_node_size;
    size_t hid_layer_size;
    size_t out_node_size;
    size_t out_layer_size;
    double learning_rate;
    ActivationType hid_act_type;
    ActivationType out_act_type;
    Layer layers[];
} Network;

/**
 * @brief Creates a dynamically-sized, 3-layer neural network.
 */
Network *nn_create(size_t inp_count, size_t hid_count, size_t out_count);

/**
 * @brief Frees the memory allocated for the neural network.
 */
void nn_free(Network *nn);

/**
 * @brief Feeds input data into the input layer of the network.
 */
void nn_feed_input(Network *nn, const Vector *v);

/**
 * @brief Performs forward propagation through the network.
 */
void nn_feed_forward(Network *nn);

/**
 * @brief Performs back propagation to adjust weights based on target classification.
 */
void nn_back_propagate(Network *nn, int target_classification);

/**
 * @brief Returns the network's classification (ID of the node with highest output).
 */
int nn_get_classification(const Network *nn);

#endif // NN_H
