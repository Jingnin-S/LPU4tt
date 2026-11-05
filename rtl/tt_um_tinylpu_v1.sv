// MIT. Standard Tiny Tapeout user-module wrapper.
module tt_um_tinylpu_v1 #(parameter integer CAPACITY=64,parameter integer LANES=1)(
    input wire[7:0] ui_in,output wire[7:0] uo_out,
    input wire[7:0] uio_in,output wire[7:0] uio_out,output wire[7:0] uio_oe,
    input wire ena,clk,rst_n
);
    tinylpu_device #(.CAPACITY(CAPACITY),.LANES(LANES)) u_device(
        .clk(clk),.rst_n(rst_n),.ena(ena),.rx_data(ui_in),.rx_valid(uio_in[0]),
        .tx_ready(uio_in[1]),.start(uio_in[7]),.tx_data(uo_out),.status(uio_out),.output_enable(uio_oe));
    wire _unused=&{1'b0,uio_in[6:2]};
endmodule
