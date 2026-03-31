/**
 * @file nn.c
 * @brief 3-layer neural network implementation
 */

#include "nn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

/**
 * @brief Internal helper to get a node from a layer by its ID.
 */
static Node *get_node(const Layer *l, size_t node_id) {
    // This assumes all nodes in the layer have the same wcount
    size_t node_size = sizeof(Node) + (l->nodes[0].wcount * sizeof(double));
    uint8_t *sbptr = (uint8_t *)l->nodes;
    sbptr += node_id * node_size;
    return (Node *)sbptr;
}

/**
 * @brief Internal helper to get a specific layer from the network.
 */
static Layer *get_layer(const Network *nn, LayerType ltype) {
    uint8_t *sbptr = (uint8_t *)nn->layers;
    switch (ltype) {
        case LAYER_INPUT:
            break;
        case LAYER_HIDDEN:
            sbptr += nn->inp_layer_size;
            break;
        case LAYER_OUTPUT:
            sbptr += nn->inp_layer_size + nn->hid_layer_size;
            break;
        default:
            return NULL;
    }
    return (Layer *)sbptr;
}

/**
 * @brief Derivative of the activation function.
 */
static double act_fct_derivative(const Network *nn, LayerType ltype, double out_val) {
    ActivationType act = (ltype == LAYER_HIDDEN) ? nn->hid_act_type : nn->out_act_type;
    if (act == ACT_TANH) {
        return 1 - pow(tanh(out_val), 2);
    }
    return out_val * (1 - out_val); // SIGMOID
}

/**
 * @brief Activation function.
 */
static double act_fct(const Network *nn, LayerType ltype, double out_val) {
    ActivationType act = (ltype == LAYER_HIDDEN) ? nn->hid_act_type : nn->out_act_type;
    if (act == ACT_TANH) {
        return tanh(out_val);
    }
    return 1.0 / (1.0 + exp(-out_val)); // SIGMOID
}

static void update_node_weights(Network *nn, LayerType ltype, size_t id, double error) {
    Layer *update_layer = get_layer(nn, ltype);
    Node *update_node = get_node(update_layer, id);
    
    Layer *prev_layer;
    size_t prev_layer_node_size;
    
    if (ltype == LAYER_HIDDEN) {
        prev_layer = get_layer(nn, LAYER_INPUT);
        prev_layer_node_size = nn->inp_node_size;
    } else {
        prev_layer = get_layer(nn, LAYER_HIDDEN);
        prev_layer_node_size = nn->hid_node_size;
    }
    
    uint8_t *sbptr = (uint8_t *)prev_layer->nodes;
    for (size_t i = 0; i < update_node->wcount; i++) {
        Node *prev_node = (Node *)sbptr;
        update_node->weights[i] += nn->learning_rate * prev_node->output * error;
        sbptr += prev_layer_node_size;
    }
    update_node->bias += nn->learning_rate * 1 * error;
}

void nn_back_propagate(Network *nn, int target_classification) {
    Layer *ol = get_layer(nn, LAYER_OUTPUT);
    Layer *hl = get_layer(nn, LAYER_HIDDEN);
    
    // BACKPROP OUTPUT LAYER
    for (size_t o = 0; o < ol->ncount; o++) {
        Node *on = get_node(ol, o);
        int target = (o == (size_t)target_classification) ? 1 : 0;
        double error_delta = target - on->output;
        double error_signal = error_delta * act_fct_derivative(nn, LAYER_OUTPUT, on->output);
        update_node_weights(nn, LAYER_OUTPUT, o, error_signal);
    }
    
    // BACKPROP HIDDEN LAYER
    for (size_t h = 0; h < hl->ncount; h++) {
        Node *hn = get_node(hl, h);
        double output_cell_error_sum = 0;
        for (size_t o = 0; o < ol->ncount; o++) {
            Node *on = get_node(ol, o);
            int target = (o == (size_t)target_classification) ? 1 : 0;
            double error_delta = target - on->output;
            double error_signal = error_delta * act_fct_derivative(nn, LAYER_OUTPUT, on->output);
            output_cell_error_sum += error_signal * on->weights[h];
        }
        double hidden_error_signal = output_cell_error_sum * act_fct_derivative(nn, LAYER_HIDDEN, hn->output);
        update_node_weights(nn, LAYER_HIDDEN, h, hidden_error_signal);
    }
}

static void calc_node_output(Network *nn, LayerType ltype, size_t id) {
    Layer *calc_layer = get_layer(nn, ltype);
    Node *calc_node = get_node(calc_layer, id);
    
    Layer *prev_layer;
    size_t prev_layer_node_size;
    
    if (ltype == LAYER_HIDDEN) {
        prev_layer = get_layer(nn, LAYER_INPUT);
        prev_layer_node_size = nn->inp_node_size;
    } else {
        prev_layer = get_layer(nn, LAYER_HIDDEN);
        prev_layer_node_size = nn->hid_node_size;
    }
    
    uint8_t *sbptr = (uint8_t *)prev_layer->nodes;
    calc_node->output = calc_node->bias;
    
    for (size_t i = 0; i < prev_layer->ncount; i++) {
        Node *prev_node = (Node *)sbptr;
        calc_node->output += prev_node->output * calc_node->weights[i];
        sbptr += prev_layer_node_size;
    }
    calc_node->output = act_fct(nn, ltype, calc_node->output);
}

static void calc_layer(Network *nn, LayerType ltype) {
    Layer *l = get_layer(nn, ltype);
    for (size_t i = 0; i < l->ncount; i++) {
        calc_node_output(nn, ltype, i);
    }
}

void nn_feed_forward(Network *nn) {
    calc_layer(nn, LAYER_HIDDEN);
    calc_layer(nn, LAYER_OUTPUT);
}

void nn_feed_input(Network *nn, const Vector *v) {
    Layer *il = get_layer(nn, LAYER_INPUT);
    Node *iln = il->nodes;
    for (size_t i = 0; i < v->size; i++) {
        iln->output = v->vals[i];
        iln++; // Input nodes have no weights, so we can just increment the pointer
    }
}

static Layer *create_layer(size_t node_count, size_t weight_count) {
    size_t node_size = sizeof(Node) + (weight_count * sizeof(double));
    Layer *l = malloc(sizeof(Layer) + (node_count * node_size));
    if (!l) return NULL;
    
    l->ncount = node_count;
    
    Node *dn = malloc(node_size);
    if (!dn) {
        free(l);
        return NULL;
    }
    
    dn->bias = 0;
    dn->output = 0;
    dn->wcount = weight_count;
    for (size_t o = 0; o < weight_count; o++) dn->weights[o] = 0;
    
    uint8_t *sbptr = (uint8_t *)l->nodes;
    for (size_t i = 0; i < node_count; i++) {
        memcpy(sbptr + (i * node_size), dn, node_size);
    }
    
    free(dn);
    return l;
}

static void init_weights(Network *nn, LayerType ltype) {
    size_t node_size = (ltype == LAYER_HIDDEN) ? nn->hid_node_size : nn->out_node_size;
    Layer *l = get_layer(nn, ltype);
    uint8_t *sbptr = (uint8_t *)l->nodes;
    
    for (size_t o = 0; o < l->ncount; o++) {
        Node *n = (Node *)sbptr;
        for (size_t i = 0; i < n->wcount; i++) {
            n->weights[i] = 0.7 * (rand() / (double)RAND_MAX);
            if (i % 2) n->weights[i] = -n->weights[i];
        }
        n->bias = rand() / (double)RAND_MAX;
        if (o % 2) n->bias = -n->bias;
        sbptr += node_size;
    }
}

Network *nn_create(size_t inp_count, size_t hid_count, size_t out_count) {
    size_t inp_node_size = sizeof(Node);
    size_t inp_layer_size = sizeof(Layer) + (inp_count * inp_node_size);
    
    size_t hid_weights_count = inp_count;
    size_t hid_node_size = sizeof(Node) + (hid_weights_count * sizeof(double));
    size_t hid_layer_size = sizeof(Layer) + (hid_count * hid_node_size);
    
    size_t out_weights_count = hid_count;
    size_t out_node_size = sizeof(Node) + (out_weights_count * sizeof(double));
    size_t out_layer_size = sizeof(Layer) + (out_count * out_node_size);
    
    Network *nn = malloc(sizeof(Network) + inp_layer_size + hid_layer_size + out_layer_size);
    if (!nn) return NULL;
    
    nn->inp_node_size = inp_node_size;
    nn->inp_layer_size = inp_layer_size;
    nn->hid_node_size = hid_node_size;
    nn->hid_layer_size = hid_layer_size;
    nn->out_node_size = out_node_size;
    nn->out_layer_size = out_layer_size;
    
    nn->hid_act_type = ACT_SIGMOID;
    nn->out_act_type = ACT_SIGMOID;
    nn->learning_rate = 0.2;
    
    // Initialize layers
    uint8_t *sbptr = (uint8_t *)nn->layers;
    
    Layer *il = create_layer(inp_count, 0);
    memcpy(sbptr, il, inp_layer_size);
    free(il);
    
    sbptr += inp_layer_size;
    Layer *hl = create_layer(hid_count, inp_count);
    memcpy(sbptr, hl, hid_layer_size);
    free(hl);
    
    sbptr += hid_layer_size;
    Layer *ol = create_layer(out_count, hid_count);
    memcpy(sbptr, ol, out_layer_size);
    free(ol);
    
    init_weights(nn, LAYER_HIDDEN);
    init_weights(nn, LAYER_OUTPUT);
    
    return nn;
}

void nn_free(Network *nn) {
    free(nn);
}

int nn_get_classification(const Network *nn) {
    Layer *l = get_layer(nn, LAYER_OUTPUT);
    double max_out = -1.0;
    int max_ind = -1;
    
    for (size_t i = 0; i < l->ncount; i++) {
        Node *on = get_node(l, i);
        if (on->output > max_out) {
            max_out = on->output;
            max_ind = (int)i;
        }
    }
    return max_ind;
}
