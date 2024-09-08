const std = @import("std");
const fs = std.fs;
const MLP_space = @import("./mlp.zig");
const MLP = MLP_space.MLP;

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

    // ### hyper parametes
    const NETWORK_STRUCTURE = [_]usize{ 28 * 28, 20, 10 };
    const START_BATCH_SIZE = 500;
    const LEARN_RATE: f64 = 0.01 / @as(f64, @floatFromInt(START_BATCH_SIZE));
    const EPOCHS = 1;

    // ### initialization
    var mlp = try MLP(&NETWORK_STRUCTURE).init(.{ .learn_rate = LEARN_RATE }, allocator);

    // ### training
    const mnist = try @import("./emnist.zig").mnist();

    var batch = BatchIterator{
        .images = undefined,
        .labels = undefined,
    };

    for (0..EPOCHS) |epoch_counter| {
        std.debug.print("{} epoch\nlearn rate {d:.5}\nbatch size: {d}\n", .{ epoch_counter, mlp.learn_rate * START_BATCH_SIZE, START_BATCH_SIZE });
        for (0..mnist.training_data.labels.len / START_BATCH_SIZE) |batch_index| {
            const img_batch = 28 * 28 * START_BATCH_SIZE;
            batch.images = mnist.training_data.data[img_batch * batch_index .. img_batch * (batch_index + 1)];
            batch.labels = mnist.training_data.labels[START_BATCH_SIZE * batch_index .. START_BATCH_SIZE * (batch_index + 1)];
            batch.index = 0;
            mlp.backprop(&batch);
        }

        batch.images = mnist.test_data.data;
        batch.labels = mnist.test_data.labels;
        batch.index = 0;

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
