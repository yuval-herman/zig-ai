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

fn cost(output: f64, expected: f64) f64 {
    const err = output - expected;
    return err * err;
}

fn costDerivative(output: f64, expected: f64) f64 {
    return 2 * (output - expected);
}

fn outputCost(output: []const f64, expected: []const f64) f64 {
    var loss: f64 = 0;
    for (output, expected) |o, e| {
        loss += cost(o, e);
    }
    return loss;
}

const HyperParameters = struct {
    learn_rate: f64,
};

pub fn MLP(NETWORK_STRUCTURE: []const usize) type {
    comptime var bias_amount = 0;
    comptime var weights_amount = 0;
    inline for (1..NETWORK_STRUCTURE.len) |i| {
        bias_amount += NETWORK_STRUCTURE[i];
        weights_amount += NETWORK_STRUCTURE[i - 1] * NETWORK_STRUCTURE[i];
    }

    return struct {
        const ThisType = @This();
        learn_rate: f64,

        biases: [bias_amount]f64 = undefined,
        bias_grads: [bias_amount]f64 = undefined,
        weights: [weights_amount]f64 = undefined,
        weights_grads: [weights_amount]f64 = undefined,

        layers_output: [NETWORK_STRUCTURE.len][]f64 = undefined,
        activated_layers_output: [NETWORK_STRUCTURE.len][]f64 = undefined,
        node_derivatives: [NETWORK_STRUCTURE.len][]f64 = undefined,

        pub fn init(parameters: HyperParameters, allocator: Allocator) !ThisType {
            var mlp = MLP(NETWORK_STRUCTURE){
                .learn_rate = parameters.learn_rate,
            };

            mlp.node_derivatives[0] = try allocator.alloc(f64, NETWORK_STRUCTURE[0]);
            inline for (1..NETWORK_STRUCTURE.len) |i| {
                mlp.activated_layers_output[i] = try allocator.alloc(f64, NETWORK_STRUCTURE[i]);
                mlp.layers_output[i] = try allocator.alloc(f64, NETWORK_STRUCTURE[i]);
                mlp.node_derivatives[i] = try allocator.alloc(f64, NETWORK_STRUCTURE[i]);
            }
            for (&mlp.biases, &mlp.bias_grads) |*bias, *grad| {
                bias.* = random.float(f64);
                grad.* = 0;
            }
            for (&mlp.weights, &mlp.weights_grads) |*weight, *grad| {
                weight.* = random.float(f64) / 100;
                grad.* = 0;
            }
            return mlp;
        }

        pub fn forward(self: *ThisType, input: []const f64) []f64 {
            assert(input.len == NETWORK_STRUCTURE[0]);
            var bias_index: usize = 0;
            var weight_index: usize = 0;
            self.activated_layers_output[0] = @constCast(input);
            self.layers_output[0] = @constCast(input);

            for (1..self.activated_layers_output.len) |layer_index| {
                for (0..self.activated_layers_output[layer_index].len) |out_index| {
                    const active_out = &self.activated_layers_output[layer_index][out_index];
                    const out = &self.layers_output[layer_index][out_index];
                    out.* = self.biases[bias_index];
                    bias_index += 1;
                    for (self.activated_layers_output[layer_index - 1]) |in| {
                        out.* += in * self.weights[weight_index];
                        weight_index += 1;
                    }
                    active_out.* = activation(out.*);
                }
            }

            return self.activated_layers_output[self.activated_layers_output.len - 1];
        }

        pub fn backprop(self: *ThisType, batch: *BatchIterator) void {
            var expected = std.mem.zeroes([10]f64);
            assert(expected.len == NETWORK_STRUCTURE[NETWORK_STRUCTURE.len - 1]);

            while (batch.next()) |data_point| {
                expected[data_point.label] = 1;
                defer expected[data_point.label] = 0;
                _ = self.forward(data_point.image);

                var bias_index: usize = bias_amount;
                var weight_index: usize = weights_amount;

                self.activated_layers_output[0] = @constCast(data_point.image);

                for (
                    self.activated_layers_output[NETWORK_STRUCTURE.len - 1],
                    self.layers_output[NETWORK_STRUCTURE.len - 1],
                    self.node_derivatives[NETWORK_STRUCTURE.len - 1],
                    expected,
                ) |a_output, output, *node_derivative, expected_value| {
                    node_derivative.* = costDerivative(a_output, expected_value) * activationDerivative(output);
                }

                var layer_index = NETWORK_STRUCTURE.len - 1;
                while (layer_index > 0) : (layer_index -= 1) {
                    @memset(self.node_derivatives[layer_index - 1], 0);
                    var out_index = NETWORK_STRUCTURE[layer_index];
                    while (out_index > 0) {
                        out_index -= 1;
                        bias_index -= 1;
                        self.bias_grads[bias_index] += self.node_derivatives[layer_index][out_index];

                        var in_index = NETWORK_STRUCTURE[layer_index - 1];
                        while (in_index > 0) {
                            in_index -= 1;
                            weight_index -= 1;
                            self.weights_grads[weight_index] +=
                                self.activated_layers_output[layer_index - 1][in_index] *
                                self.node_derivatives[layer_index][out_index];
                            self.node_derivatives[layer_index - 1][in_index] += self.weights[weight_index] *
                                self.node_derivatives[layer_index][out_index];
                        }
                    }
                    for (self.node_derivatives[layer_index - 1], self.layers_output[layer_index - 1]) |*node_derivative, z| {
                        node_derivative.* *= activationDerivative(z);
                    }
                }
            }

            for (&self.weights, &self.weights_grads) |*weight, *grad| {
                weight.* -= grad.* * self.learn_rate;
                grad.* = 0;
            }
            for (&self.biases, &self.bias_grads) |*bias, *grad| {
                bias.* -= grad.* * self.learn_rate;
                grad.* = 0;
            }
        }
    };
}

pub const BatchIterator = struct {
    images: []const f64,
    labels: []const u8,
    index: usize = 0,

    pub fn next(self: *BatchIterator) ?struct { image: []const f64, label: u8 } {
        defer self.index += 1;
        if (self.index == self.labels.len) {
            return null;
        }
        return .{
            .image = self.images[self.index * 28 * 28 .. (self.index + 1) * 28 * 28],
            .label = self.labels[self.index],
        };
    }

    fn reset(self: *BatchIterator) *BatchIterator {
        self.index = 0;
        return self;
    }
};

test "forward mlp test" {
    var arena = ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var mlp = try MLP(&.{ 2, 2, 2 }).init(.{ .learn_rate = 1 }, arena.allocator());

    const input = [_]f64{ 2, 2 };

    const o1 = activation(input[0] * mlp.weights[0] + input[1] * mlp.weights[1] + mlp.biases[0]);
    const o2 = activation(input[0] * mlp.weights[2] + input[1] * mlp.weights[3] + mlp.biases[1]);
    const o11 = activation(o1 * mlp.weights[4] + o2 * mlp.weights[5] + mlp.biases[2]);
    const o22 = activation(o1 * mlp.weights[6] + o2 * mlp.weights[7] + mlp.biases[3]);

    const out = [_]f64{ o11, o22 };
    try std.testing.expectEqualSlices(f64, &out, mlp.forward(&input));
}

test outputCost {
    const loss = outputCost(&[2]f64{ 0.5, 2 }, &[2]f64{ 1, 1 });
    try std.testing.expectEqual(1.25, loss);
}

test "backpropagation mlp test" {
    var arena = ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var mlp = try MLP(&.{ 2, 2, 2 }).init(.{ .learn_rate = 1 }, arena.allocator());

    const input = [_]f64{ 2, 2 };

    const z1 = input[0] * mlp.weights[0] + input[1] * mlp.weights[1] + mlp.biases[0];
    const a1 = activation(z1);
    const z2 = input[0] * mlp.weights[2] + input[1] * mlp.weights[3] + mlp.biases[1];
    const a2 = activation(z2);
    const z11 = a1 * mlp.weights[4] + a2 * mlp.weights[5] + mlp.biases[2];
    const a11 = activation(z11);
    const z22 = a1 * mlp.weights[6] + a2 * mlp.weights[7] + mlp.biases[3];
    const a22 = activation(z22);

    const node_derivatives = [_]f64{
        activationDerivative(z11) * costDerivative(a11, 2),
        activationDerivative(z22) * costDerivative(a22, 2),
    };

    const wg0 =
        input[0] *
        activationDerivative(z1) *
        (mlp.weights[4] * node_derivatives[0] +
        mlp.weights[6] * node_derivatives[1]);
    const wg4 = a1 * node_derivatives[0];

    mlp.backprop(&input, &.{ 2, 2 });
    try std.testing.expectEqual(wg0, mlp.weights_grads[0]);
    try std.testing.expectEqual(wg4, mlp.weights_grads[4]);
}
