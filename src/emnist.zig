const std = @import("std");

fn Data(Dtype: type, Ltype: type) type {
    return struct {
        training_data: struct { data: []Dtype, labels: []Ltype },
        test_data: struct { data: []Dtype, labels: []Ltype },
    };
}

pub fn balanced_dataset(allocator: std.mem.Allocator) !Data(f64, u8) {
    var data = Data(f64, u8){
        .test_data = undefined,
        .training_data = undefined,
    };
    data.test_data.labels = try readIdxFile("./emnist/emnist-balanced-test-labels-idx1-ubyte.gz", allocator);
    data.training_data.labels = try readIdxFile("./emnist/emnist-balanced-train-labels-idx1-ubyte.gz", allocator);

    {
        const raw = try readIdxFile("./emnist/emnist-balanced-test-images-idx3-ubyte.gz", allocator);
        defer allocator.free(raw);
        data.test_data.data = try convertToFloats(raw, allocator);
    }

    {
        const raw = try readIdxFile("./emnist/emnist-balanced-train-images-idx3-ubyte.gz", allocator);
        defer allocator.free(raw);
        data.training_data.data = try convertToFloats(raw, allocator);
    }

    return data;
}

pub fn digits_dataset(allocator: std.mem.Allocator) !Data(f64, u8) {
    var data = Data(f64, u8){
        .test_data = undefined,
        .training_data = undefined,
    };
    data.test_data.labels = try readIdxFile("./emnist/emnist-digits-test-labels-idx1-ubyte.gz", allocator);
    data.training_data.labels = try readIdxFile("./emnist/emnist-digits-train-labels-idx1-ubyte.gz", allocator);

    {
        const raw = try readIdxFile("./emnist/emnist-digits-test-images-idx3-ubyte.gz", allocator);
        defer allocator.free(raw);
        data.test_data.data = try convertToFloats(raw, allocator);
    }

    {
        const raw = try readIdxFile("./emnist/emnist-digits-train-images-idx3-ubyte.gz", allocator);
        defer allocator.free(raw);
        data.training_data.data = try convertToFloats(raw, allocator);
    }

    return data;
}

fn convertToFloats(buffer: []u8, allocator: std.mem.Allocator) ![]f64 {
    const conv: []f64 = try allocator.alloc(f64, buffer.len);
    for (conv, buffer) |*in, mn| {
        in.* = @as(f64, @floatFromInt(mn)) / 255;
    }
    return conv;
}

fn readIdxFile(path: []const u8, allocator: std.mem.Allocator) ![]u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var decomp = std.compress.gzip.decompressor(file.reader());
    var idx: usize = 0;
    const magic_number = try decomp.get(4);

    const data_type_size: u8 = switch (magic_number[2]) {
        0x08, 0x09 => 1,
        0x0B => 2,
        0x0C => 4,
        0x0D => 4,
        0x0E => 8,
        else => unreachable,
    };
    const dimensions: u8 = magic_number[3];

    // std.debug.print("magic number: {x}\n", .{magic_number});
    // std.debug.print("data is a {} dimension matrix of {} bytes\n", .{ dimensions, data_type_size });

    var data_size_len: usize = data_type_size;
    for (0..dimensions) |_| {
        const d_raw = try decomp.get(4);
        const d_size = std.mem.bigToNative(u32, std.mem.bytesToValue(u32, d_raw));
        // std.debug.print("size in {} dimension is {}\n", .{ d, d_size });
        data_size_len *= d_size;
    }
    // std.debug.print("overall data size is {}\n", .{data_size_len});
    const data = try allocator.alloc(u8, data_size_len);
    while (try decomp.next()) |d| {
        @memcpy(data[idx .. idx + d.len], d);
        idx += d.len;
    }
    return data;
}
