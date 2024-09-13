const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const assert = std.debug.assert;

var xoshiro256 = std.Random.DefaultPrng.init(0);
const random = xoshiro256.random();

fn activation(x: f64) f64 {
    return if (x > 0) x else 0.01 * x;
}

fn activationDerivative(x: f64) f64 {
    return if (x > 0) 1 else 0.01;
}

pub fn cost(output: f64, expected: f64) f64 {
    const err = output - expected;
    return err * err;
}

fn costDerivative(output: f64, expected: f64) f64 {
    return 2 * (output - expected);
}

const HyperParameters = struct {
    learn_rate: f64,
};

pub fn MLP(NETWORK_STRUCTURE: []const u32) type {
    comptime var bias_amount = 0;
    comptime var weights_amount = 0;
    inline for (1..NETWORK_STRUCTURE.len) |i| {
        bias_amount += NETWORK_STRUCTURE[i];
        weights_amount += NETWORK_STRUCTURE[i - 1] * NETWORK_STRUCTURE[i];
    }

    return struct {
        const ThisType = @This();
        learn_rate: f64,

        biases: []f64 = undefined,
        bias_grads: []f64 = undefined,
        weights: []f64 = undefined,
        weights_grads: []f64 = undefined,

        layers_output: [NETWORK_STRUCTURE.len][]f64 = undefined,
        activated_layers_output: [NETWORK_STRUCTURE.len][]f64 = undefined,
        node_derivatives: [NETWORK_STRUCTURE.len][]f64 = undefined,

        pub fn init(parameters: HyperParameters, allocator: Allocator) !ThisType {
            var mlp = MLP(NETWORK_STRUCTURE){
                .learn_rate = parameters.learn_rate,
            };

            mlp.node_derivatives[0] = try allocator.alloc(f64, NETWORK_STRUCTURE[0]);
            // layers output uses the first opinter to point to the input and thus does not need allocated space
            inline for (1..NETWORK_STRUCTURE.len) |i| {
                mlp.layers_output[i] = try allocator.alloc(f64, NETWORK_STRUCTURE[i]);
                mlp.activated_layers_output[i] = try allocator.alloc(f64, NETWORK_STRUCTURE[i]);
                mlp.node_derivatives[i] = try allocator.alloc(f64, NETWORK_STRUCTURE[i]);
            }

            mlp.biases = try allocator.alloc(f64, bias_amount);
            mlp.bias_grads = try allocator.alloc(f64, bias_amount);
            mlp.weights = try allocator.alloc(f64, weights_amount);
            mlp.weights_grads = try allocator.alloc(f64, weights_amount);

            for (mlp.biases, mlp.bias_grads) |*bias, *grad| {
                bias.* = random.float(f64) / 100;
                grad.* = 0;
            }
            for (mlp.weights, mlp.weights_grads) |*weight, *grad| {
                weight.* = random.float(f64) / 1000;
                grad.* = 0;
            }

            return mlp;
        }

        // `self: *const ThisType` found to be faster then `self: ThisType` after profiling
        fn inner_forward(self: *const ThisType, layers_output: [][]f64, activated_layers_output: [][]f64, input: []const f64) []f64 {
            assert(input.len == NETWORK_STRUCTURE[0]);
            var bias_index: usize = 0;
            var weight_index: usize = 0;
            activated_layers_output[0] = @constCast(input);
            layers_output[0] = @constCast(input);

            inline for (1..NETWORK_STRUCTURE.len) |layer_index| {
                for (0..NETWORK_STRUCTURE[layer_index]) |out_index| {
                    const out = &layers_output[layer_index][out_index];
                    out.* = self.biases[bias_index];
                    bias_index += 1;

                    const vec_size = NETWORK_STRUCTURE[layer_index - 1];
                    const vec_type = @Vector(vec_size, f64);

                    const inputs: vec_type = activated_layers_output[layer_index - 1][0..vec_size].*;
                    const weights: vec_type = self.weights[weight_index..][0..vec_size].*;
                    weight_index += vec_size;
                    out.* += @reduce(.Add, inputs * weights);

                    activated_layers_output[layer_index][out_index] = activation(out.*);
                }
            }

            return activated_layers_output[activated_layers_output.len - 1];
        }

        pub fn forward(self: *ThisType, input: []const f64) []f64 {
            return self.inner_forward(&self.layers_output, &self.activated_layers_output, input);
        }

        pub fn backprop(self: *ThisType, batch: anytype) void {
            self.extra_backprop(self.bias_grads, self.weights_grads, &self.layers_output, &self.activated_layers_output, batch);
            self.applyGrads();
        }

        pub fn extra_backprop(
            self: *const ThisType,
            bias_grads: []f64,
            weights_grads: []f64,
            layers_output: [][]f64,
            activated_layers_output: [][]f64,
            node_derivatives: [][]f64,
            batch: anytype,
        ) void {
            while (batch.next()) |data_point| {
                _ = self.inner_forward(layers_output, activated_layers_output, data_point.input);

                var bias_index: usize = bias_amount;
                var weight_index: usize = weights_amount;

                activated_layers_output[0] = @constCast(data_point.input);

                for (
                    activated_layers_output[NETWORK_STRUCTURE.len - 1],
                    layers_output[NETWORK_STRUCTURE.len - 1],
                    node_derivatives[NETWORK_STRUCTURE.len - 1],
                    data_point.output,
                ) |a_output, output, *node_derivative, expected_value| {
                    node_derivative.* = costDerivative(a_output, expected_value) * activationDerivative(output);
                }
                comptime var layer_index = NETWORK_STRUCTURE.len - 1;
                inline while (layer_index > 0) : (layer_index -= 1) {
                    @memset(node_derivatives[layer_index - 1], 0);
                    var out_index = NETWORK_STRUCTURE[layer_index];
                    while (out_index > 0) {
                        out_index -= 1;
                        bias_index -= 1;
                        bias_grads[bias_index] += node_derivatives[layer_index][out_index];

                        const vec_size = NETWORK_STRUCTURE[layer_index - 1];
                        const vec_type = @Vector(vec_size, f64);

                        var grads_vec: vec_type = weights_grads[weight_index - vec_size ..][0..vec_size].*;
                        const activated_layers_output_vec: vec_type = activated_layers_output[layer_index - 1][0..vec_size].*;
                        const node_derivative: vec_type = @splat(node_derivatives[layer_index][out_index]);

                        grads_vec += activated_layers_output_vec * node_derivative;
                        weights_grads[weight_index - vec_size ..][0..vec_size].* = grads_vec;

                        var prev_node_derivatives: vec_type = node_derivatives[layer_index - 1][0..vec_size].*;
                        const weights: vec_type = self.weights[weight_index - vec_size ..][0..vec_size].*;
                        prev_node_derivatives += weights * node_derivative;
                        node_derivatives[layer_index - 1][0..vec_size].* = prev_node_derivatives;
                        weight_index -= vec_size;
                    }
                    for (node_derivatives[layer_index - 1], layers_output[layer_index - 1]) |*node_derivative, z| {
                        node_derivative.* *= activationDerivative(z);
                    }
                }
            }
        }
        pub fn applyGrads(self: *ThisType) void {
            for (self.weights, self.weights_grads) |*weight, *grad| {
                weight.* -= grad.* * self.learn_rate;
                grad.* = 0;
            }
            for (self.biases, self.bias_grads) |*bias, *grad| {
                bias.* -= grad.* * self.learn_rate;
                grad.* = 0;
            }
        }
    };
}
