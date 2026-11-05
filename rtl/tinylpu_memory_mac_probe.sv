// Tiny-LPU architecture probes.
//
// These are deliberately small, synthesizable resource probes rather than a
// complete accelerator.  The byte arrays expose registered read ports and a
// single byte write.  The MAC exposes one registered product stage followed
// by a registered accumulator stage.  No timing or area result from this file
// should be interpreted as a physical Fmax claim.

module tinylpu_flat_byte_array_probe #(
    parameter integer CAPACITY = 32,
    parameter integer READ_PORTS = 1,
    parameter integer ADDR_BITS = (CAPACITY <= 1) ? 1 : $clog2(CAPACITY)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire wr_en,
    input  wire [ADDR_BITS-1:0] wr_addr,
    input  wire [7:0] wr_data,
    input  wire [READ_PORTS*ADDR_BITS-1:0] rd_addr,
    output reg  [READ_PORTS*8-1:0] rd_data
);
    reg [7:0] mem [0:CAPACITY-1];
    integer port;

    always @(posedge clk) begin
        if (!rst_n) begin
            rd_data <= {READ_PORTS*8{1'b0}};
        end else begin
            if (wr_en && (wr_addr < CAPACITY)) begin
                mem[wr_addr] <= wr_data;
            end
            for (port = 0; port < READ_PORTS; port = port + 1) begin
                if (rd_addr[port*ADDR_BITS +: ADDR_BITS] < CAPACITY) begin
                    rd_data[port*8 +: 8] <=
                        mem[rd_addr[port*ADDR_BITS +: ADDR_BITS]];
                end else begin
                    rd_data[port*8 +: 8] <= 8'h00;
                end
            end
        end
    end
endmodule


module tinylpu_banked_byte_array_probe #(
    parameter integer CAPACITY = 32,
    parameter integer BANKS = 2,
    parameter integer ADDR_BITS = (CAPACITY <= 1) ? 1 : $clog2(CAPACITY)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire wr_en,
    input  wire [ADDR_BITS-1:0] wr_addr,
    input  wire [7:0] wr_data,
    input  wire [BANKS*ADDR_BITS-1:0] rd_addr,
    output reg  [BANKS*8-1:0] rd_data
);
    localparam integer BANK_DEPTH = (CAPACITY + BANKS - 1) / BANKS;
    reg [7:0] bank_mem [0:BANKS-1][0:BANK_DEPTH-1];
    integer port;
    integer write_bank;
    integer write_index;
    integer read_address;

    always @(posedge clk) begin
        if (!rst_n) begin
            rd_data <= {BANKS*8{1'b0}};
        end else begin
            if (wr_en && (wr_addr < CAPACITY)) begin
                write_bank = wr_addr % BANKS;
                write_index = wr_addr / BANKS;
                bank_mem[write_bank][write_index] <= wr_data;
            end
            for (port = 0; port < BANKS; port = port + 1) begin
                read_address = rd_addr[port*ADDR_BITS +: ADDR_BITS];
                // Port number is the physical bank number.  A request for a
                // different bank is not silently routed here; a real gather
                // arbiter/crossbar is a separate resource not included in
                // this native one-read-per-bank probe.
                if (read_address < CAPACITY &&
                    ((read_address % BANKS) == port)) begin
                    rd_data[port*8 +: 8] <=
                        bank_mem[port][read_address / BANKS];
                end else begin
                    rd_data[port*8 +: 8] <= 8'h00;
                end
            end
        end
    end
endmodule


module tinylpu_int4_mac_probe #(
    parameter integer LANES = 1,
    parameter integer ACC_BITS = 16,
    parameter integer DSP_DEPTH = 3
) (
    input  wire clk,
    input  wire rst_n,
    input  wire clear_acc,
    input  wire issue_valid,
    input  wire issue_last,
    input  wire [LANES*4-1:0] a_in,
    input  wire [LANES*4-1:0] w_in,
    output reg  done,
    output wire [LANES*ACC_BITS-1:0] acc_out
);
    localparam integer PIPE_STAGES = DSP_DEPTH - 1;
    reg signed [7:0] product_pipe [0:PIPE_STAGES-1][0:LANES-1];
    reg valid_pipe [0:PIPE_STAGES-1];
    reg last_pipe [0:PIPE_STAGES-1];
    reg signed [ACC_BITS-1:0] acc_reg [0:LANES-1];
    integer stage;
    integer lane;

    genvar out_lane;
    generate
        for (out_lane = 0; out_lane < LANES; out_lane = out_lane + 1) begin : GEN_ACC_OUT
            assign acc_out[out_lane*ACC_BITS +: ACC_BITS] = acc_reg[out_lane];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            done <= 1'b0;
            for (stage = 0; stage < PIPE_STAGES; stage = stage + 1) begin
                valid_pipe[stage] <= 1'b0;
                last_pipe[stage] <= 1'b0;
                for (lane = 0; lane < LANES; lane = lane + 1) begin
                    product_pipe[stage][lane] <= 8'sd0;
                end
            end
            for (lane = 0; lane < LANES; lane = lane + 1) begin
                acc_reg[lane] <= {ACC_BITS{1'b0}};
            end
        end else begin
            done <= 1'b0;

            // Commit the product after DSP_DEPTH-1 registered pipeline
            // stages.  The old final stage is observed before the shift, so
            // an issue at edge t commits at edge t+DSP_DEPTH-1.
            if (valid_pipe[PIPE_STAGES-1]) begin
                for (lane = 0; lane < LANES; lane = lane + 1) begin
                    acc_reg[lane] <= acc_reg[lane] +
                                     product_pipe[PIPE_STAGES-1][lane];
                end
                if (last_pipe[PIPE_STAGES-1]) begin
                    done <= 1'b1;
                end
            end

            if (clear_acc) begin
                for (lane = 0; lane < LANES; lane = lane + 1) begin
                    acc_reg[lane] <= {ACC_BITS{1'b0}};
                end
            end

            for (stage = PIPE_STAGES - 1; stage > 0; stage = stage - 1) begin
                valid_pipe[stage] <= valid_pipe[stage-1];
                last_pipe[stage] <= last_pipe[stage-1];
                for (lane = 0; lane < LANES; lane = lane + 1) begin
                    product_pipe[stage][lane] <= product_pipe[stage-1][lane];
                end
            end
            valid_pipe[0] <= issue_valid;
            last_pipe[0] <= issue_valid && issue_last;
            if (issue_valid) begin
                for (lane = 0; lane < LANES; lane = lane + 1) begin
                    product_pipe[0][lane] <=
                        $signed(a_in[lane*4 +: 4]) *
                        $signed(w_in[lane*4 +: 4]);
                end
            end
        end
    end
endmodule
