// MIT. Byte-stream INT4/ACC32 controller and datapath.
module tinylpu_device #(parameter integer CAPACITY=128,parameter integer LANES=2)(
    input wire clk,rst_n,ena,
    input wire[7:0] rx_data,input wire rx_valid,tx_ready,start,
    output wire[7:0] tx_data,output wire[7:0] status,output wire[7:0] output_enable
);
    localparam integer I_IDLE=0,I_HEADER=1,I_ENDS=2,I_CHECK=3,I_LOAD=4,
        I_STORE_REQ=5,I_STORE_RESP=6,I_TX=7,I_INIT=8,I_SEED_REQ=9,I_SEED_RESP=10,
        I_CORE=11,I_WB=12,I_DONE=13,I_ERROR=14,I_MVM_VALIDATE=15;
    localparam[15:0] IDLE=16'b1<<I_IDLE,HEADER=16'b1<<I_HEADER,
        ENDS=16'b1<<I_ENDS,CHECK=16'b1<<I_CHECK,LOAD=16'b1<<I_LOAD,
        STORE_REQ=16'b1<<I_STORE_REQ,STORE_RESP=16'b1<<I_STORE_RESP,
        TX=16'b1<<I_TX,INIT=16'b1<<I_INIT,SEED_REQ=16'b1<<I_SEED_REQ,
        SEED_RESP=16'b1<<I_SEED_RESP,CORE=16'b1<<I_CORE,WB=16'b1<<I_WB,
        DONE=16'b1<<I_DONE,ERROR=16'b1<<I_ERROR,
        MVM_VALIDATE=16'b1<<I_MVM_VALIDATE;
    reg[15:0] state;
    reg mvm_bad_latched;
    reg previous_start;
    reg[63:0] header;
    reg[2:0] header_index;
    reg[7:0] byte_index,tx_byte,w_group;
    reg[7:0] transfer_pointer,seed_pointer,wb_pointer;
    reg wb_narrow;
    reg[7:0] core_a,core_w0,core_w1;
    reg same_bank_aw0,same_bank_aw1,same_bank_w01;
    reg same_byte_aw0,same_byte_aw1;
    reg[4:0] group_row;
    reg[1:0] active_count;
    reg seed_lane;
    reg[1:0] seed_byte;
    reg[5:0] req_k;
    reg[2:0] issued;
    reg fetch_done;
    reg[12:0] a_end,w_end,y_read_end,y_write_end;
    reg[31:0] acc0,acc1;
    reg signed[7:0] product00,product01,product10,product11;
    reg pvalid0,pvalid1,plast0,plast1;
    reg[7:0] operands[0:1][0:2];
    reg[2:0] operand_mask[0:1];
    reg[11:0] response_roles;
    reg response_slot,response_parity,response_last;
    wire start_edge=start && !previous_start;
    wire selected=ena && rst_n;
    wire rx_ready=selected && !start_edge && (state[I_HEADER] || state[I_LOAD]);
    wire tx_valid=selected && !start_edge && state[I_TX];
    wire busy=!(state[I_IDLE] || state[I_DONE] || state[I_ERROR]);
    assign tx_data=tx_valid?tx_byte:8'b0;
    assign status=selected?{1'b0,state[I_ERROR],state[I_DONE],busy,tx_valid,rx_ready,2'b0}:8'b0;
    assign output_enable=selected?8'h7c:8'b0;
    wire accepted_start=start_edge && (state[I_IDLE] || state[I_DONE]);
    wire clear_valid=!selected || accepted_start;
    wire[7:0] op=header[7:0],flags=header[15:8],ha=header[23:16],hw=header[31:24];
    wire[7:0] hy=header[39:32],hm=header[47:40],hk=header[55:48],stride=header[63:56];
    wire[4:0] rows={1'b0,hm[3:0]}+5'd1;
    wire[6:0] row_bytes=({1'b0,hk[5:0]}+7'd2)>>1;
    wire[2:0] output_bytes=wb_narrow?3'd2:3'd4;
    wire[2:0] needed=active_count==2?3'b111:3'b011;
    wire[12:0] y_end=y_read_end>y_write_end?y_read_end:y_write_end;
    wire[12:0] transfer_end={5'b0,ha}+{5'b0,hw}+13'd1;
    // For LOAD/STORE, ha is address, hw is count-minus-one.
    wire aliases=( {5'b0,hy}<a_end && {5'b0,ha}<y_end ) ||
                 ( {5'b0,hy}<w_end && {5'b0,hw}<y_end );
    // Full end-address checks precede access; these are validated byte pointers.
    wire[7:0] store_address=transfer_pointer;
    wire[7:0] seed_address=seed_pointer;
    reg bad_header;
    wire bad_mvm_header=flags[7:2]!=0 || hm[7:4]!=0 || hk[7:6]!=0 ||
        {1'b0,stride}<{2'b0,row_bytes} || a_end>13'(CAPACITY) || w_end>13'(CAPACITY) ||
        y_end>13'(CAPACITY) || aliases || (flags[1]?hy[0]!=0:hy[1:0]!=0) ||
        (flags[0] && hy[1:0]!=0);
    reg bad_read;
    wire fault=(state[I_CHECK] && bad_header)||bad_read;
    reg[3:0] read_enable;
    reg[31:0] read_address;
    wire[31:0] read_data;
    wire[3:0] read_valid;
    reg write_enable;
    reg[7:0] write_address,write_data;
    reg[11:0] request_roles;
    wire request_slot=req_k[0];
    wire request_parity=req_k[0];
    wire request_last=req_k==hk[5:0];
    reg[2:0] issued_next;
    reg[2:0] core_grant;
    reg[7:0] role_address[0:2];
    wire[7:0] init_w1=w_group+stride;
    reg[7:0] merged[0:2];
    reg[2:0] merged_mask;
    reg[31:0] wb_value;
    reg[1:0] wb_part;

    function automatic signed[3:0] nibble(input[7:0] value,input parity);
        begin nibble=parity?value[7:4]:value[3:0];end
    endfunction
    function automatic[31:0] saturated(input[31:0] value,input narrow);
        begin
            if(!narrow)saturated=value;
            else if($signed(value)>32'sd32767)saturated=32'h00007fff;
            else if($signed(value)<-32'sd32768)saturated=32'hffff8000;
            else saturated=value;
        end
    endfunction

    always @*begin
        bad_header=1'b0;
        case(op)
          8'd1,8'd2:bad_header=flags!=0 || transfer_end>13'(CAPACITY);
          8'd15:bad_header=flags!=0 || ha!=0 || hw!=0;
          8'd3:bad_header=mvm_bad_latched;
          default:bad_header=1'b1;
        endcase
        bad_read=1'b0;
        if(state[I_STORE_RESP])bad_read=!read_valid[store_address[1:0]];
        if(state[I_SEED_RESP])bad_read=!read_valid[seed_address[1:0]];
        if(state[I_CORE])for(integer b=0;b<4;b=b+1)
            if(response_roles[b*3+:3]!=0 && !read_valid[b])bad_read=1'b1;
    end

    // Parallel per-role grants avoid serial dynamic updates of the bank mux.
    // A wins over W0, then W1; identical addresses broadcast in one request.
    always @*begin
        read_enable=0;read_address=0;request_roles=0;
        role_address[0]=core_a;
        role_address[1]=core_w0;
        role_address[2]=core_w1;
        core_grant=needed&~issued;
        if(core_grant[0] && same_bank_aw0 && !same_byte_aw0)core_grant[1]=0;
        if((core_grant[0] && same_bank_aw1 && !same_byte_aw1) ||
           (core_grant[1] && same_bank_w01))core_grant[2]=0;
        for(integer b=0;b<4;b=b+1)begin
            unique case(1'b1)
              state[I_STORE_REQ]:if(store_address[1:0]==2'(b))begin
                  read_enable[b]=1;read_address[b*8+:8]=store_address;
              end
              state[I_SEED_REQ]:if(seed_address[1:0]==2'(b))begin
                  read_enable[b]=1;read_address[b*8+:8]=seed_address;
              end
              state[I_CORE]:if(!fetch_done)begin
                  for(integer r=0;r<3;r=r+1)
                      request_roles[b*3+r]=core_grant[r] && role_address[r][1:0]==2'(b);
                  read_enable[b]=|request_roles[b*3+:3];
                  if(request_roles[b*3])read_address[b*8+:8]=role_address[0][7:0];
                  else if(request_roles[b*3+1])read_address[b*8+:8]=role_address[1][7:0];
                  else if(request_roles[b*3+2])read_address[b*8+:8]=role_address[2][7:0];
              end
              default:begin end
            endcase
        end
        issued_next=issued|core_grant;
        // Error handling discards any in-flight reads. They have no architectural
        // side effect, so a validity fault does not feed back into the read mux.
        if(!selected || start_edge)begin read_enable=0;read_address=0;request_roles=0;end
    end

    always @*begin
        for(integer r=0;r<3;r=r+1)merged[r]=operands[response_slot][r];
        merged_mask=operand_mask[response_slot];
        for(integer b=0;b<4;b=b+1)for(integer r=0;r<3;r=r+1)
            if(response_roles[b*3+r])begin
                merged[r]=read_data[b*8+:8];merged_mask[r]=1'b1;
            end
    end
    always @*begin
        write_enable=0;write_address=0;write_data=0;
        wb_part=wb_narrow?{1'b0,byte_index[0]}:byte_index[1:0];
        wb_value=saturated((wb_narrow?byte_index[1]:byte_index[2])?acc1:acc0,wb_narrow);
        if(state[I_LOAD] && rx_valid)begin
            write_enable=1'b1;write_address=transfer_pointer;write_data=rx_data;
        end
        if(state[I_WB])begin
            write_enable=1'b1;
            write_address=wb_pointer;
            write_data=wb_value[wb_part*8+:8];
        end
        // Fault states cannot write: LOAD/WB are disjoint from CHECK/read responses.
        if(!selected || start_edge)write_enable=1'b0;
    end

    tinylpu_device_memory #(.CAPACITY(CAPACITY)) memory(
        .clk(clk),.clear_valid(clear_valid),.wr_en(write_enable),.wr_addr(write_address),.wr_data(write_data),
        .rd_en(read_enable),.rd_addr(read_address),.rd_data(read_data),.rd_valid(read_valid));

    always @(posedge clk)begin
        if(!rst_n || !ena)begin
            state<=IDLE;previous_start<=0;header<=0;header_index<=0;byte_index<=0;tx_byte<=0;
            mvm_bad_latched<=0;
            transfer_pointer<=0;seed_pointer<=0;wb_pointer<=0;wb_narrow<=0;
            group_row<=0;w_group<=0;active_count<=0;seed_lane<=0;seed_byte<=0;
            req_k<=0;issued<=0;fetch_done<=0;
            core_a<=0;core_w0<=0;core_w1<=0;
            same_bank_aw0<=0;same_bank_aw1<=0;same_bank_w01<=0;
            same_byte_aw0<=0;same_byte_aw1<=0;
            a_end<=0;w_end<=0;y_read_end<=0;y_write_end<=0;
            acc0<=0;acc1<=0;pvalid0<=0;pvalid1<=0;plast0<=0;plast1<=0;
            product00<=0;product01<=0;product10<=0;product11<=0;
            response_roles<=0;response_slot<=0;response_parity<=0;response_last<=0;
            operand_mask[0]<=0;operand_mask[1]<=0;
            for(integer s=0;s<2;s=s+1)for(integer r=0;r<3;r=r+1)operands[s][r]<=0;
        end else if(start_edge)begin
            previous_start<=1;response_roles<=0;pvalid0<=0;pvalid1<=0;
            if(state[I_IDLE] || state[I_DONE])begin
                state<=HEADER;header<=0;header_index<=0;byte_index<=0;tx_byte<=0;
                group_row<=0;w_group<=0;active_count<=0;seed_lane<=0;seed_byte<=0;
                req_k<=0;issued<=0;fetch_done<=0;acc0<=0;acc1<=0;
                operand_mask[0]<=0;operand_mask[1]<=0;
            end else state<=ERROR;
        end else if(fault)begin
            previous_start<=start;state<=ERROR;response_roles<=0;pvalid0<=0;pvalid1<=0;
        end else begin
            previous_start<=start;
            response_roles<=request_roles;response_slot<=request_slot;
            response_parity<=request_parity;response_last<=request_last;
            unique case(1'b1)
              state[I_HEADER]:if(rx_valid)begin
                  if(header_index==0)header<={56'b0,rx_data};
                  else header[header_index*8+:8]<=rx_data;
                  if(header_index==(op==3?3'd7:3'd3))begin state<=ENDS;header_index<=0;end
                  else header_index<=header_index+1'b1;
              end
              state[I_ENDS]:begin
                  a_end<={5'b0,ha}+{6'b0,row_bytes};
                  w_end<={5'b0,hw}+({9'b0,hm[3:0]}*{5'b0,stride})+{6'b0,row_bytes};
                  y_read_end<={5'b0,hy}+(flags[0]?({8'b0,rows}<<2):13'b0);
                  y_write_end<={5'b0,hy}+(flags[1]?({8'b0,rows}<<1):({8'b0,rows}<<2));
                  state<=op==3?MVM_VALIDATE:CHECK;
              end
              state[I_MVM_VALIDATE]:begin
                  mvm_bad_latched<=bad_mvm_header;
                  state<=CHECK;
              end
              state[I_CHECK]:begin
                  byte_index<=0;transfer_pointer<=ha;
                  case(op)
                    1:state<=LOAD;
                    2:state<=STORE_REQ;
                    3:begin group_row<=0;w_group<=hw;state<=INIT;end
                    15:state<=DONE;
                    default:state<=ERROR;
                  endcase
              end
              state[I_LOAD]:if(rx_valid)begin
                  if(byte_index==hw)begin state<=HEADER;header_index<=0;end
                  else begin byte_index<=byte_index+1'b1;transfer_pointer<=transfer_pointer+1'b1;end
              end
              state[I_STORE_REQ]:state<=STORE_RESP;
              state[I_STORE_RESP]:begin tx_byte<=read_data[store_address[1:0]*8+:8];state<=TX;end
              state[I_TX]:if(tx_ready)begin
                  if(byte_index==hw)begin state<=HEADER;header_index<=0;end
                  else begin byte_index<=byte_index+1'b1;transfer_pointer<=transfer_pointer+1'b1;state<=STORE_REQ;end
              end
              state[I_INIT]:begin
                  active_count<=(LANES==2 && rows-group_row>1)?2'd2:2'd1;
                  seed_pointer<=hy+({3'b0,group_row}<<2);
                  wb_pointer<=hy+(flags[1]?({3'b0,group_row}<<1):({3'b0,group_row}<<2));
                  wb_narrow<=flags[1];
                  core_a<=ha;core_w0<=w_group;core_w1<=init_w1;
                  // All active addresses advance by the same byte offset. Checked
                  // ranges prevent wrap, so these relations hold for the group.
                  same_bank_aw0<=ha[1:0]==w_group[1:0];
                  same_bank_aw1<=ha[1:0]==init_w1[1:0];
                  same_bank_w01<=stride[1:0]==0;
                  same_byte_aw0<=ha==w_group;same_byte_aw1<=ha==init_w1;
                  acc0<=0;acc1<=0;pvalid0<=0;pvalid1<=0;operand_mask[0]<=0;operand_mask[1]<=0;
                  req_k<=0;issued<=0;seed_lane<=0;seed_byte<=0;fetch_done<=0;
                  for(integer s=0;s<2;s=s+1)for(integer r=0;r<3;r=r+1)operands[s][r]<=0;
                  state<=flags[0]?SEED_REQ:CORE;
              end
              state[I_SEED_REQ]:state<=SEED_RESP;
              state[I_SEED_RESP]:begin
                  if(seed_lane)acc1[seed_byte*8+:8]<=read_data[seed_address[1:0]*8+:8];
                  else acc0[seed_byte*8+:8]<=read_data[seed_address[1:0]*8+:8];
                  if(seed_byte==3)begin
                      seed_byte<=0;
                      if({1'b0,seed_lane}+2'd1==active_count)state<=CORE;
                      else begin seed_lane<=1;seed_pointer<=seed_pointer+1'b1;state<=SEED_REQ;end
                  end else begin seed_byte<=seed_byte+1'b1;seed_pointer<=seed_pointer+1'b1;state<=SEED_REQ;end
              end
              state[I_CORE]:begin
                  pvalid1<=pvalid0;plast1<=plast0;product10<=product00;product11<=product01;pvalid0<=0;
                  if(pvalid1)begin
                      acc0<=acc0+{{24{product10[7]}},product10};
                      if(active_count==2)acc1<=acc1+{{24{product11[7]}},product11};
                      if(plast1)begin state<=WB;byte_index<=0;end
                  end
                  if(|response_roles)begin
                      for(integer r=0;r<3;r=r+1)operands[response_slot][r]<=merged[r];
                      operand_mask[response_slot]<=merged_mask;
                      if((merged_mask&needed)==needed)begin
                          pvalid0<=1;plast0<=response_last;
                          product00<=nibble(merged[0],response_parity)*nibble(merged[1],response_parity);
                          product01<=active_count==2?nibble(merged[0],response_parity)*nibble(merged[2],response_parity):8'sd0;
                      end
                  end
                  if(!fetch_done)begin
                      if(issued==0)operand_mask[request_slot]<=0;
                      if(issued_next==needed)begin
                          issued<=0;
                          if(req_k==hk[5:0])fetch_done<=1;
                          else begin
                              req_k<=req_k+1'b1;
                              if(req_k[0])begin
                                  core_a<=core_a+1'b1;core_w0<=core_w0+1'b1;core_w1<=core_w1+1'b1;
                              end
                          end
                      end else issued<=issued_next;
                  end
              end
              state[I_WB]:begin
                  if({1'b0,byte_index}+9'd1==active_count*output_bytes)begin
                      if(group_row+{3'b0,active_count}==rows)begin state<=HEADER;header_index<=0;end
                      else begin group_row<=group_row+5'(LANES);w_group<=w_group+8'(LANES)*stride;state<=INIT;end
                  end else begin byte_index<=byte_index+1'b1;wb_pointer<=wb_pointer+1'b1;end
              end
              default:begin end
            endcase
        end
    end
endmodule

