const std = @import("std");
const fs = std.fs;
const MLP = @import("./mlp.zig").MLP;
const Thread = std.Thread;

const BatchIterator = struct {
    images: []const f64,
    labels: []const u8,
    index: usize = 0,
    output_array: [10]f64 = std.mem.zeroes([10]f64),

    pub fn next(self: *BatchIterator) ?struct { input: []const f64, output: []const f64 } {
        if (self.index == self.labels.len) {
            return null;
        }
        defer self.index += 1;

        self.output_array[self.labels[self.index]] = 1;
        if (self.index != 0) self.output_array[self.labels[self.index - 1]] = 0;

        return .{
            .input = self.images[self.index * 28 * 28 .. (self.index + 1) * 28 * 28],
            .output = &self.output_array,
        };
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var threadPool = Thread.Pool{ .allocator = allocator, .threads = &[_]std.Thread{} };
    var waitgroup = Thread.WaitGroup{};
    defer threadPool.deinit();
    const threads_amount = (Thread.getCpuCount() catch 1);

    try threadPool.init(.{
        .allocator = allocator,
        .n_jobs = @intCast(threads_amount),
    });

    // ### hyper parametes
    const NETWORK_STRUCTURE = [_]u32{ 28 * 28, 200, 10 };
    const BATCH_SIZE = 500;
    const LEARN_RATE: f64 = 0.5 / @as(f64, @floatFromInt(BATCH_SIZE));
    const EPOCHS = 5;

    // ### initialization
    const NetType = MLP(&NETWORK_STRUCTURE);
    var mlp = try NetType.init(.{ .learn_rate = LEARN_RATE }, allocator);

    // ### training
    const mnist = try @import("./emnist.zig").mnist();
    const ThreadData = struct {
        bias_grads: []f64,
        weights_grads: []f64,
        layers_output: [][]f64,
        activated_layers_output: [][]f64,
        node_derivatives: [][]f64,
        batch: BatchIterator,
    };
    std.debug.print("threads: {}\n", .{threads_amount});
    const threads_data = try allocator.alloc(ThreadData, threads_amount);
    for (threads_data) |*td| {
        td.bias_grads = try allocator.alloc(f64, mlp.bias_grads.len);
        td.weights_grads = try allocator.alloc(f64, mlp.weights_grads.len);
        td.layers_output = try allocator.alloc([]f64, mlp.layers_output.len);
        td.activated_layers_output = try allocator.alloc([]f64, mlp.activated_layers_output.len);
        td.node_derivatives = try allocator.alloc([]f64, mlp.node_derivatives.len);
        td.node_derivatives[0] = try allocator.alloc(f64, mlp.node_derivatives[0].len);
        for (1..NETWORK_STRUCTURE.len) |i| {
            td.layers_output[i] = try allocator.alloc(f64, mlp.layers_output[i].len);
            td.activated_layers_output[i] = try allocator.alloc(f64, mlp.activated_layers_output[i].len);
            td.node_derivatives[i] = try allocator.alloc(f64, mlp.node_derivatives[i].len);
        }
    }

    for (0..EPOCHS) |epoch_counter| {
        mlp.learn_rate = @max(
            (-0.00015 * @as(f32, @floatFromInt(epoch_counter))) + LEARN_RATE,
            0.0001 / @as(comptime_float, @floatFromInt(BATCH_SIZE)),
        );
        std.debug.print("{} epoch\nlearn rate {d:.5}\nbatch size: {d}\n", .{ epoch_counter, mlp.learn_rate * BATCH_SIZE, BATCH_SIZE });

        for (0..mnist.training_data.labels.len / BATCH_SIZE) |batch_index| {
            const img_size = 28 * 28;
            const base_chunk_size = @divFloor(BATCH_SIZE, threads_amount);
            var reminder: usize = BATCH_SIZE - base_chunk_size * threads_amount;

            var start: usize = batch_index * BATCH_SIZE;
            var end: usize = 0;
            waitgroup.reset();
            for (threads_data) |*td| {
                end = start + base_chunk_size;
                if (reminder != 0) {
                    reminder -= 1;
                    end += 1;
                }

                td.batch.images = mnist.training_data.data[img_size * start .. img_size * end];
                td.batch.labels = mnist.training_data.labels[start..end];
                td.batch.index = 0;

                threadPool.spawnWg(&waitgroup, &NetType.extra_backprop, .{
                    &mlp,
                    td.bias_grads,
                    td.weights_grads,
                    td.layers_output,
                    td.activated_layers_output,
                    td.node_derivatives,
                    &td.batch,
                });
                start = end;
            }
            waitgroup.wait();
            for (threads_data) |td| {
                for (
                    mlp.bias_grads,
                    td.bias_grads,
                ) |*bias_grad, *td_bias_grad| {
                    bias_grad.* += td_bias_grad.*;
                    td_bias_grad.* = 0;
                }
                for (mlp.weights_grads, td.weights_grads) |*weight_grad, *td_weights_grad| {
                    weight_grad.* += td_weights_grad.*;
                    td_weights_grad.* = 0;
                }
            }

            mlp.applyGrads();
        }

        var batch = BatchIterator{
            .images = mnist.test_data.data,
            .labels = mnist.test_data.labels,
            .index = 0,
        };

        var correct: u16 = 0;
        var wrong: u16 = 0;

        while (batch.next()) |data_point| {
            const output = mlp.forward(data_point.input);
            var max = output[0];
            var max_index: usize = 0;
            for (output[1..], 1..) |o, i| {
                if (o > max) {
                    max_index = i;
                    max = o;
                }
            }

            // I use batch.index-1 because batch.next() incerements the index after the function returns
            if (max_index == batch.labels[batch.index - 1]) {
                correct += 1;
            } else {
                wrong += 1;
            }
        }
        std.debug.print("correct: {}\nwrong: {}\ncorrect percent: {d}%\n\n", .{
            correct,
            wrong,
            100 * @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(correct + wrong)),
        });
    }
    try writeWeights(NETWORK_STRUCTURE, mlp);
}

fn writeMnistJson(mnist: anytype) !void {
    const out_file = try fs.cwd().createFile("./mnist.json", .{});
    defer out_file.close();

    var writer = std.io.bufferedWriter(out_file.writer());

    _ = try writer.write("{");
    _ = try writer.write("\"data\": [");
    try writer.flush();
    for (mnist.test_data.data[0 .. mnist.test_data.data.len - 1]) |data| {
        try std.fmt.format(writer.writer(), "{d},", .{data});
    }
    try std.fmt.format(writer.writer(), "{d}", .{mnist.test_data.data[mnist.test_data.data.len - 1]});
    _ = try writer.write("],");
    _ = try writer.write("\"label\": [");
    try writer.flush();
    for (mnist.test_data.labels[0 .. mnist.test_data.labels.len - 1]) |label| {
        try std.fmt.format(writer.writer(), "{d},", .{label});
    }
    try std.fmt.format(writer.writer(), "{d}", .{mnist.test_data.labels[mnist.test_data.labels.len - 1]});
    _ = try writer.write("]}");
    try writer.flush();
}

fn writeWeights(NETWORK_STRUCTURE: anytype, mlp: anytype) !void {
    const out_file = try fs.cwd().createFile("./weights.json", .{});
    defer out_file.close();

    var writer = std.io.bufferedWriter(out_file.writer());

    _ = try writer.write("{");
    _ = try writer.write("\"structure\": [");
    try writer.flush();
    for (NETWORK_STRUCTURE[0 .. NETWORK_STRUCTURE.len - 1]) |layer_size| {
        try std.fmt.format(writer.writer(), "{d},", .{layer_size});
    }
    try std.fmt.format(writer.writer(), "{d}", .{NETWORK_STRUCTURE[NETWORK_STRUCTURE.len - 1]});
    _ = try writer.write("],");
    _ = try writer.write("\"biases\": [");
    try writer.flush();
    for (mlp.biases[0 .. mlp.biases.len - 1]) |bias| {
        try std.fmt.format(writer.writer(), "{d},", .{bias});
    }
    try std.fmt.format(writer.writer(), "{d}", .{mlp.biases[mlp.biases.len - 1]});
    _ = try writer.write("],");
    _ = try writer.write("\"weights\": [");
    try writer.flush();
    for (mlp.weights[0 .. mlp.weights.len - 1]) |weight| {
        try std.fmt.format(writer.writer(), "{d},", .{weight});
    }
    try std.fmt.format(writer.writer(), "{d}", .{mlp.weights[mlp.weights.len - 1]});
    _ = try writer.write("]}");
    try writer.flush();
}
