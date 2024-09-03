const std = @import("std");
const fs = std.fs;
const MLP_space = @import("./mlp.zig");
const MLP = MLP_space.MLP;
const BatchIterator = MLP_space.BatchIterator;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // ### hyper parametes
    const NETWORK_STRUCTURE = [_]usize{ 28 * 28, 50, 10 };
    const START_BATCH_SIZE = 5;
    const LEARN_RATE: f64 = 0.5 / @as(f64, @floatFromInt(START_BATCH_SIZE));
    const EPOCHS = 20;

    // ### initialization
    var mlp = try MLP(&NETWORK_STRUCTURE).init(.{ .learn_rate = LEARN_RATE }, allocator);

    // ### training
    const mnsit = try @import("./emnist.zig").mnist();

    // {
    //     const out_file = try fs.cwd().createFile("./mnist.json", .{});
    //     defer out_file.close();

    //     var writer = std.io.bufferedWriter(out_file.writer());

    //     _ = try writer.write("{");
    //     _ = try writer.write("\"data\": [");
    //     try writer.flush();
    //     for (mnsit.test_data.data[0 .. mnsit.test_data.data.len - 1]) |data| {
    //         try std.fmt.format(writer.writer(), "{d},", .{data});
    //     }
    //     try std.fmt.format(writer.writer(), "{d}", .{mnsit.test_data.data[mnsit.test_data.data.len - 1]});
    //     _ = try writer.write("],");
    //     _ = try writer.write("\"label\": [");
    //     try writer.flush();
    //     for (mnsit.test_data.labels[0 .. mnsit.test_data.labels.len - 1]) |label| {
    //         try std.fmt.format(writer.writer(), "{d},", .{label});
    //     }
    //     try std.fmt.format(writer.writer(), "{d}", .{mnsit.test_data.labels[mnsit.test_data.labels.len - 1]});
    //     _ = try writer.write("]}");
    //     try writer.flush();
    // }

    var batch = BatchIterator{
        .images = undefined,
        .labels = undefined,
    };
    // var expected = std.mem.zeroes([10]f64);

    var batch_size: usize = START_BATCH_SIZE;
    const max_batch_size_epoch = 10;
    const max_batch_size = mnsit.training_data.labels.len / 1000;
    const b_const = (max_batch_size - START_BATCH_SIZE) / ((max_batch_size_epoch) * (max_batch_size_epoch));
    for (0..EPOCHS) |epoch_counter| {
        mlp.learn_rate = LEARN_RATE * @as(f64, @floatFromInt(batch_size));
        batch_size = @min(max_batch_size - 1, b_const * epoch_counter * epoch_counter + START_BATCH_SIZE);
        mlp.learn_rate = LEARN_RATE / @as(f64, @floatFromInt(batch_size));

        std.debug.print("{} epoch\nlearn rate {d:.5}\nbatch size: {d}\n", .{ epoch_counter, mlp.learn_rate * @as(f64, @floatFromInt(batch_size)), batch_size });
        for (0..mnsit.training_data.labels.len / batch_size) |batch_index| {
            const img_batch = 28 * 28 * batch_size;
            batch.images = mnsit.training_data.data[img_batch * batch_index .. img_batch * (batch_index + 1)];
            batch.labels = mnsit.training_data.labels[batch_size * batch_index .. batch_size * (batch_index + 1)];
            batch.index = 0;
            mlp.backprop(&batch);
        }

        // expected[batch.labels[batch_size - 1]] = 1;
        // defer expected[batch.labels[batch_size - 1]] = 0;

        // std.debug.print("cost {d}\n", .{outputCost(
        //     mlp.activated_layers_output[mlp.activated_layers_output.len - 1],
        //     &expected,
        // )});

        batch.images = mnsit.test_data.data;
        batch.labels = mnsit.test_data.labels;
        batch.index = 0;

        var correct: u16 = 0;
        var wrong: u16 = 0;
        while (batch.next()) |data_point| {
            const output = mlp.forward(data_point.image);
            var max = output[0];
            var max_index: usize = 0;
            for (output[1..], 1..) |o, i| {
                if (o > max) {
                    max_index = i;
                    max = o;
                }
            }

            if (max_index == data_point.label) {
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
