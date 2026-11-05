`timescale 1ns/1ps
// SPDX-License-Identifier: MIT
// Functional/cycle gate simulation; actual timing comes from extracted STA.
module tb_tinylpu_gate_vectors;
    reg clk=0;
    always #50 clk=~clk;
    reg rst_n=0,ena=0;
    reg[7:0] ui_in=0,uio_in=0;
    wire[7:0] uo_out,uio_out,uio_oe;
    supply1 VPWR;
    supply0 VGND;
    tt_um_tinylpu_v1 dut(.*);
    integer file,rc,cycle=0;
    reg[31:0] inputs;
    reg[23:0] expected;
    reg[4095:0] path;
    initial begin
        if(!$value$plusargs("VECTORS=%s",path))$fatal(1,"Missing VECTORS");
        file=$fopen(path,"r");if(!file)$fatal(1,"Cannot open vectors");
        while(!$feof(file))begin
            rc=$fscanf(file,"%h %h\n",inputs,expected);
            if(rc==2)begin
                @(negedge clk);
                ui_in=inputs[7:0];uio_in={inputs[10],5'b0,inputs[9],inputs[8]};
                rst_n=inputs[11];ena=inputs[12];
                @(posedge clk);#20;
                if({uio_oe,uio_out,uo_out}!==expected)begin
                    $display("GATE mismatch cycle=%0d inputs=%h actual=%h expected=%h",cycle,inputs,{uio_oe,uio_out,uo_out},expected);
                    $fatal(1,"Gate cycle mismatch");
                end
                cycle=cycle+1;
            end
        end
        $display("DEVICE GATE VECTORS PASS cycles=%0d",cycle);
        $fclose(file);$finish(0);
    end
endmodule
