const std = @import("std");

fn Data(Dtype: type, Ltype: type) type {
    return struct {
        training_data: struct { data: []Dtype, labels: []Ltype },
        test_data: struct { data: []Dtype, labels: []Ltype },
    };
}
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();
pub fn mnist() !Data(f64, u8) {
    var data = Data(f64, u8){
        .test_data = undefined,
        .training_data = undefined,
    };
    // ### Labels
    {
        var buf = try allocator.alloc(u8, 40008);
        _ = try readComp("./emnist/emnist-digits-test-labels-idx1-ubyte.gz", buf);
        data.test_data.labels = buf[4 * 2 ..]; // test_labels 1
    }
    {
        var buf = try allocator.alloc(u8, 240008);
        _ = try readComp("./emnist/emnist-digits-train-labels-idx1-ubyte.gz", buf);
        data.training_data.labels = buf[4 * 2 ..]; // train_labels 1
    }
    // ### Images
    {
        const buf: []u8 = try allocator.alloc(u8, 31360016);
        defer allocator.free(buf);
        _ = try readComp("./emnist/emnist-digits-test-images-idx3-ubyte.gz", buf);
        const conv: []f64 = try allocator.alloc(f64, 31360016 - (4 * 4));
        for (conv, buf[4 * 4 ..]) |*in, mn| {
            in.* = @as(f64, @floatFromInt(mn)) / 255;
        }
        data.test_data.data = conv; // test_images 28*28
    }
    {
        const buf: []u8 = try allocator.alloc(u8, 188160016);
        defer allocator.free(buf);
        _ = try readComp("./emnist/emnist-digits-train-images-idx3-ubyte.gz", buf);
        const conv: []f64 = try allocator.alloc(f64, 188160016 - (4 * 4));
        for (conv, buf[4 * 4 ..]) |*in, mn| {
            in.* = @as(f64, @floatFromInt(mn)) / 255;
        }
        data.training_data.data = conv; // train_images 28*28
    }
    return data;
}

fn readComp(path: []const u8, buffer: []u8) !usize {
    const data = try std.fs.cwd().openFile(path, .{});
    defer data.close();
    var decomp = std.compress.gzip.decompressor(data.reader());
    var idx: usize = 0;
    while (try decomp.next()) |d| {
        // std.debug.print("{x}\n", .{d});
        @memcpy(buffer[idx .. idx + d.len], d);
        idx += d.len;
    }
    return idx;
}
