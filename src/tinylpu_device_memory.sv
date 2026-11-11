// MIT. Complete byte-stream INT4/ACC32 device. One clock, no instruction RAM.
module tinylpu_device_memory #(parameter integer CAPACITY=128)(
    input wire clk, input wire clear_valid,
    // One write per bank and clock. wr_word is the shared address[7:2]:
    // LOAD enables one bank, result writeback up to all four.
    input wire[3:0] wr_en,input wire[5:0] wr_word,input wire[31:0] wr_data,
    input wire[3:0] rd_en,input wire[31:0] rd_addr,
    output wire[31:0] rd_data,output wire[3:0] rd_valid
);
    localparam integer DEPTH=CAPACITY/4;
    localparam integer INDEX_BITS=$clog2(DEPTH);
    genvar b;
    generate for(b=0;b<4;b=b+1)begin:bank
        reg[7:0] data[0:DEPTH-1];
        reg[DEPTH-1:0] valid;
        reg[7:0] q;
        reg qv;
        wire[7:0] address=rd_addr[b*8+:8];
        assign rd_data[b*8+:8]=q;
        assign rd_valid[b]=qv;
        always @(posedge clk)begin
            if(clear_valid)begin valid<={DEPTH{1'b0}};q<=0;qv<=0;end
            else begin
                if(wr_en[b] && {1'b0,wr_word,2'(b)}<9'(CAPACITY))begin
                    data[wr_word[INDEX_BITS-1:0]]<=wr_data[b*8+:8];valid[wr_word[INDEX_BITS-1:0]]<=1'b1;
                end
                if(rd_en[b])begin
                    if(address[1:0]==2'(b) && {1'b0,address}<9'(CAPACITY))begin
                        q<=data[address[INDEX_BITS+1:2]];qv<=valid[address[INDEX_BITS+1:2]];
                    end else begin q<=0;qv<=0;end
                end
            end
        end
    end endgenerate
endmodule

