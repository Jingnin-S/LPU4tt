`timescale 1ns/1ps
// MIT. Pin-only functional gate wrapper with explicit constant supplies.
module tb_tinylpu_gate;
    reg clk,rst_n,ena;
    reg[7:0] ui_in,uio_in;
    wire[7:0] uo_out,uio_out,uio_oe;
    supply1 VPWR;
    supply0 VGND;
    tt_um_tinylpu_v1 dut(.*);
endmodule
