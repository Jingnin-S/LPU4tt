`timescale 1ns/1ps

module tb_tinylpu_memory_mac_probe;
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst_n = 1'b0;

    reg flat_wr_en = 1'b0;
    reg [4:0] flat_wr_addr = 5'd0;
    reg [7:0] flat_wr_data = 8'd0;
    reg [4:0] flat_rd_addr = 5'd0;
    wire [7:0] flat_rd_data;
    tinylpu_flat_byte_array_probe #(.CAPACITY(32), .READ_PORTS(1)) flat1 (
        .clk(clk), .rst_n(rst_n), .wr_en(flat_wr_en), .wr_addr(flat_wr_addr),
        .wr_data(flat_wr_data), .rd_addr(flat_rd_addr), .rd_data(flat_rd_data));

    reg [5:0] bank_wr_addr = 6'd0;
    reg bank_wr_en = 1'b0;
    reg [7:0] bank_wr_data = 8'd0;
    reg [11:0] bank_rd_addr = 12'd0;
    wire [15:0] bank_rd_data;
    tinylpu_banked_byte_array_probe #(.CAPACITY(64), .BANKS(2)) bank2 (
        .clk(clk), .rst_n(rst_n), .wr_en(bank_wr_en), .wr_addr(bank_wr_addr),
        .wr_data(bank_wr_data), .rd_addr(bank_rd_addr), .rd_data(bank_rd_data));

    reg clear_acc = 1'b0;
    reg issue_valid = 1'b0;
    reg issue_last = 1'b0;
    reg [7:0] a_in = 8'd0;
    reg [7:0] w_in = 8'd0;
    wire done_d2;
    wire done_d3;
    wire [31:0] acc_out_d2;
    wire [31:0] acc_out_d3;
    tinylpu_int4_mac_probe #(.LANES(2), .ACC_BITS(16), .DSP_DEPTH(2)) mac2 (
        .clk(clk), .rst_n(rst_n), .clear_acc(clear_acc),
        .issue_valid(issue_valid), .issue_last(issue_last), .a_in(a_in),
        .w_in(w_in), .done(done_d2), .acc_out(acc_out_d2));
    tinylpu_int4_mac_probe #(.LANES(2), .ACC_BITS(16), .DSP_DEPTH(3)) mac3 (
        .clk(clk), .rst_n(rst_n), .clear_acc(clear_acc),
        .issue_valid(issue_valid), .issue_last(issue_last), .a_in(a_in),
        .w_in(w_in), .done(done_d3), .acc_out(acc_out_d3));

    task expect_equal;
        input [31:0] actual;
        input [31:0] expected;
        input [1023:0] label;
        begin
            if (actual !== expected) begin
                $display("FAIL %0s actual=%h expected=%h", label, actual, expected);
                $fatal(1, "RTL probe mismatch: %0s", label);
            end
        end
    endtask

    initial begin
        // Reset, write one byte, then verify the registered flat read.
        #12;
        rst_n = 1'b1;
        @(negedge clk);
        flat_wr_en = 1'b1;
        flat_wr_addr = 5'd7;
        flat_wr_data = 8'h5a;
        @(negedge clk);
        flat_wr_en = 1'b0;
        flat_rd_addr = 5'd7;
        @(negedge clk);
        #1;
        expect_equal(flat_rd_data, 8'h5a, "flat registered read");

        // A banked write/read pair: addresses 2 and 3 use different banks.
        bank_wr_en = 1'b1;
        bank_wr_addr = 6'd2;
        bank_wr_data = 8'ha5;
        @(negedge clk);
        bank_wr_addr = 6'd3;
        bank_wr_data = 8'h3c;
        @(negedge clk);
        bank_wr_en = 1'b0;
        bank_rd_addr[5:0] = 6'd2;
        bank_rd_addr[11:6] = 6'd3;
        @(negedge clk);
        #1;
        expect_equal(bank_rd_data, 16'h3ca5, "banked registered reads");

        // Two lanes: 2*-3=-6 and 7*-4=-28, with the final issue drained by
        // the registered product stage before done is asserted.
        clear_acc = 1'b1;
        @(negedge clk);
        clear_acc = 1'b0;
        a_in = {4'h7, 4'h2};
        w_in = {4'hc, 4'hd};
        issue_valid = 1'b1;
        issue_last = 1'b1;
        @(negedge clk);
        issue_valid = 1'b0;
        issue_last = 1'b0;
        @(negedge clk);
        #1;
        expect_equal(done_d2, 1'b1, "DSP_DEPTH=2 done latency");
        expect_equal(done_d3, 1'b0, "DSP_DEPTH=3 remains in flight");
        expect_equal(acc_out_d2, 32'hffe4fffa, "DSP_DEPTH=2 lane results");
        @(negedge clk);
        #1;
        expect_equal(done_d3, 1'b1, "DSP_DEPTH=3 done latency");
        expect_equal(acc_out_d3, 32'hffe4fffa, "DSP_DEPTH=3 lane results");
        $display("RTL probe simulation passed");
        $finish(0);
    end
endmodule
