#include "tinylpu_device.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
enum State {Idle, Header, Ends, Check, Load, StoreRequest, StoreResponse,
            Transmit, GroupInit, SeedRequest, SeedResponse, Core, Writeback, Done, Error, MvmValidate};
struct Request {
  std::array<bool,4> enable{};
  std::array<unsigned,4> address{}, roles{};
  unsigned slot=0, parity=0; bool last=false;
};
struct Product {bool valid=false,last=false; std::array<int,2> value{};};
std::int64_t signed32(std::uint32_t value) {
  return value & 0x80000000u ? static_cast<std::int64_t>(value)-4294967296LL : value;
}
int nibble(unsigned value,unsigned parity) {
  int n=static_cast<int>((value>>(4*parity))&15u);return n>=8?n-16:n;
}
struct Device {
  unsigned capacity,lanes;
  State state=Idle;
  bool previous_start=false;
  std::array<unsigned,8> h{};
  unsigned header_index=0,byte_index=0,group=0,active=0,seed_lane=0,seed_byte=0;
  unsigned req_k=0,issued=0; bool fetch_done=false;
  unsigned a_end=0,w_end=0,y_read_end=0,y_write_end=0;
  bool mvm_bad=false;
  unsigned tx=0,event=0;
  std::vector<unsigned> memory;
  std::vector<bool> valid;
  std::array<unsigned,4> q{};
  std::array<bool,4> qvalid{};
  Request pending;
  std::array<std::array<unsigned,3>,2> buffer{};
  std::array<unsigned,2> mask{},acc{};
  std::array<Product,2> pipe{};
  std::uint64_t cycles=0,commands=0,macs=0,loads=0,stores=0;
  std::string debug;
  Device(unsigned c,unsigned l):capacity(c),lanes(l),memory(c),valid(c,false) {
    if(c<32||c>256||c%4||(l!=1&&l!=2))throw std::invalid_argument("capacity32..256 multiple4, lanes1/2 required");
  }
  unsigned rows() const{return h[5]+1;}
  unsigned k() const{return h[6]+1;}
  unsigned row_bytes() const{return (k()+1)/2;}
  unsigned out_bytes() const{return h[1]&2u?2u:4u;}
  bool start_edge(unsigned in) const{return (in&(1u<<10))&&!previous_start;}
  unsigned output(unsigned in) const {
    if(!(in&(1u<<11))||!(in&(1u<<12)))return 0;
    bool edge=start_edge(in);
    unsigned io=0;
    if((state==Header||state==Load)&&!edge)io|=4;
    if(state==Transmit&&!edge)io|=8;
    if(state!=Idle&&state!=Done&&state!=Error)io|=16;
    if(state==Done)io|=32;
    if(state==Error)io|=64;
    return ((state==Transmit&&!edge)?tx:0)|(io<<8)|(0x7cu<<16);
  }
  Request request() const {
    Request r;
    if(state==StoreRequest||state==SeedRequest) {
      unsigned address=state==StoreRequest?h[2]+byte_index:h[4]+(group+seed_lane)*4+seed_byte;
      r.enable[address%4]=true;r.address[address%4]=address;
    } else if(state==Core&&!fetch_done) {
      std::array<unsigned,3> addresses={h[2]+req_k/2,h[3]+group*h[7]+req_k/2,h[3]+(group+1)*h[7]+req_k/2};
      unsigned needed=active==2?7u:3u;
      for(unsigned role=0;role<3;++role) {
        if(!(needed&(1u<<role))||(issued&(1u<<role)))continue;
        unsigned bank=addresses[role]%4;
        if(!r.enable[bank]){r.enable[bank]=true;r.address[bank]=addresses[role];}
        if(r.address[bank]==addresses[role])r.roles[bank]|=1u<<role;
      }
      r.slot=req_k%2;r.parity=req_k%2;r.last=req_k==h[6];
    }
    return r;
  }
  bool bad_header() const {
    if(h[0]==1||h[0]==2)return h[1]!=0||h[2]+h[3]+1>capacity;
    if(h[0]==15)return h[1]||h[2]||h[3];
    if(h[0]!=3)return true;
    unsigned yend=std::max(y_read_end,y_write_end);
    bool alias=(h[4]<a_end&&h[2]<yend)||(h[4]<w_end&&h[3]<yend);
    return (h[1]&0xfcu)||h[5]>15||h[6]>63||h[7]<row_bytes()||
           a_end>capacity||w_end>capacity||yend>capacity||alias||
           h[4]%out_bytes()!=0||((h[1]&1u)&&h[4]%4!=0);
  }
  void reset() {
    state=Idle;previous_start=false;h={};header_index=byte_index=group=active=seed_lane=seed_byte=0;
    req_k=issued=0;fetch_done=false;a_end=w_end=y_read_end=y_write_end=tx=0;
    std::fill(valid.begin(),valid.end(),false);q={};qvalid={};pending={};buffer={};mask={};acc={};pipe={};
    commands=macs=loads=stores=0;mvm_bad=false;
  }
  unsigned write_data() const {
    unsigned unit=out_bytes(),lane=byte_index/unit,part=byte_index%unit;
    std::int64_t value=signed32(acc[lane]);
    if(unit==2)value=std::max<std::int64_t>(-32768,std::min<std::int64_t>(32767,value));
    return (static_cast<std::uint32_t>(value)>>(8*part))&255;
  }
  unsigned tick(unsigned in) {
    ++cycles;event=0;
    unsigned before=output(in);
    bool rxvalid=in&(1u<<8),ready=in&(1u<<9),edge=start_edge(in);
    bool accepted_rx=(before&(4u<<8))&&rxvalid;
    bool accepted_tx=(before&(8u<<8))&&ready;
    Request r=request();
    bool write=false,clear=false;unsigned address=0,data=0;
    if(!(in&(1u<<11))||!(in&(1u<<12))) {
      reset();clear=true;r={};
    } else if(edge) {
      previous_start=true;r={};pending={};pipe={};
      if(state==Idle||state==Done){reset();previous_start=true;state=Header;clear=true;}
      else state=Error;
    } else {
      previous_start=(in&(1u<<10))!=0;
      bool bad=(state==Check&&(h[0]==3?mvm_bad:bad_header()));
      if(state==StoreResponse)bad=bad||!qvalid[(h[2]+byte_index)%4];
      if(state==SeedResponse)bad=bad||!qvalid[(h[4]+(group+seed_lane)*4+seed_byte)%4];
      if(state==Core)for(unsigned b=0;b<4;++b)if(pending.roles[b]&&!qvalid[b])bad=true;
      if(bad){state=Error;r={};pending={};pipe={};}
      else switch(state) {
        case Idle:case Done:case Error:break;
        case Header:
          if(accepted_rx){
            event|=1;
            if(header_index==0)h={};
            h[header_index]=in&255;
            if(header_index==(h[0]==3?7u:3u)){state=Ends;header_index=0;}
            else ++header_index;
          }break;
        case Ends:
          a_end=h[2]+row_bytes();w_end=h[3]+h[5]*h[7]+row_bytes();
          y_read_end=h[4]+((h[1]&1u)?rows()*4:0);y_write_end=h[4]+rows()*out_bytes();
          state=h[0]==3?MvmValidate:Check;break;
        case MvmValidate:
          mvm_bad=bad_header();state=Check;break;
        case Check:
          ++commands;byte_index=0;
          if(h[0]==1)state=Load;
          if(h[0]==2)state=StoreRequest;
          if(h[0]==3){group=0;state=GroupInit;}
          if(h[0]==15)state=Done;
          break;
        case Load:
          if(accepted_rx){event|=1;write=true;address=h[2]+byte_index;data=in&255;++loads;
            if(byte_index==h[3]){state=Header;header_index=0;}else ++byte_index;
          }break;
        case StoreRequest:state=StoreResponse;break;
        case StoreResponse:tx=q[(h[2]+byte_index)%4];state=Transmit;break;
        case Transmit:
          if(accepted_tx){event|=2|(tx<<8);++stores;
            if(byte_index==h[3]){state=Header;header_index=0;}else{++byte_index;state=StoreRequest;}
          }break;
        case GroupInit:
          active=std::min(lanes,rows()-group);acc={};pipe={};mask={};buffer={};
          req_k=issued=seed_lane=seed_byte=0;fetch_done=false;
          state=(h[1]&1u)?SeedRequest:Core;break;
        case SeedRequest:state=SeedResponse;break;
        case SeedResponse:{
          unsigned value=q[(h[4]+(group+seed_lane)*4+seed_byte)%4];
          acc[seed_lane]=(acc[seed_lane]&~(255u<<(8*seed_byte)))|(value<<(8*seed_byte));
          if(seed_byte==3){seed_byte=0;
            if(seed_lane+1==active)state=Core;else{++seed_lane;state=SeedRequest;}
          }else{++seed_byte;state=SeedRequest;}
          break;}
        case Core:{
          Product commit=pipe[1];pipe[1]=pipe[0];pipe[0]={};
          if(commit.valid){for(unsigned lane=0;lane<active;++lane)acc[lane]+=static_cast<std::uint32_t>(commit.value[lane]);macs+=active;
            if(commit.last){state=Writeback;byte_index=0;}
          }
          unsigned response_roles=0;
          for(unsigned b=0;b<4;++b)response_roles|=pending.roles[b];
          if(response_roles){
            unsigned slot=pending.slot,newmask=mask[slot];
            for(unsigned b=0;b<4;++b)for(unsigned role=0;role<3;++role)
              if(pending.roles[b]&(1u<<role)){buffer[slot][role]=q[b];newmask|=1u<<role;}
            mask[slot]=newmask;
            if((newmask&(active==2?7u:3u))==(active==2?7u:3u)){
              pipe[0].valid=true;pipe[0].last=pending.last;
              for(unsigned lane=0;lane<active;++lane)pipe[0].value[lane]=nibble(buffer[slot][0],pending.parity)*nibble(buffer[slot][lane+1],pending.parity);
            }
          }
          if(!fetch_done){
            if(issued==0)mask[r.slot]=0;
            unsigned combined=issued;for(unsigned b=0;b<4;++b)combined|=r.roles[b];
            if(combined==(active==2?7u:3u)){issued=0;if(req_k==h[6])fetch_done=true;else ++req_k;}
            else issued=combined;
          }break;}
        case Writeback:
          write=true;address=h[4]+group*out_bytes()+byte_index;data=write_data();
          if(byte_index+1==active*out_bytes()){
            if(group+active==rows()){state=Header;header_index=0;}
            else{group+=lanes;state=GroupInit;}
          }else ++byte_index;
          break;
      }
    }
    // Separate registered memory: responses use the pre-edge data/validity.
    if(clear){std::fill(valid.begin(),valid.end(),false);q={};qvalid={};}
    else {
      for(unsigned b=0;b<4;++b)if(r.enable[b]){
        if(r.address[b]>=capacity)throw std::logic_error("unchecked read address");
        q[b]=memory[r.address[b]];qvalid[b]=valid[r.address[b]];
      }
      if(write){if(address>=capacity)throw std::logic_error("unchecked write address");memory[address]=data;valid[address]=true;}
    }
    pending=r;
    return output(in);
  }
};
}
extern "C" void *tlpu_device_create(unsigned c,unsigned l){try{error.clear();if(c<32||c>256||c%4||(l!=1&&l!=2))throw std::invalid_argument("invalid device profile");return new Device(c,l);}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" std::uint32_t tlpu_device_tick(void*p,std::uint32_t in){try{if(!p)throw std::invalid_argument("null device");return static_cast<Device*>(p)->tick(in);}catch(const std::exception&e){error=e.what();return 0xffffffffu;}}
extern "C" std::uint32_t tlpu_device_peek(void*p,std::uint32_t in){return p?static_cast<Device*>(p)->output(in):0;}
extern "C" std::uint32_t tlpu_device_events(void*p){return p?static_cast<Device*>(p)->event:0;}
extern "C" const char *tlpu_device_debug(void*p){if(!p)return "{}";auto&d=*static_cast<Device*>(p);std::ostringstream s;s<<"{\"state\":"<<d.state<<",\"cycles\":"<<d.cycles<<",\"commands\":"<<d.commands<<",\"macs\":"<<d.macs<<",\"load_bytes\":"<<d.loads<<",\"store_bytes\":"<<d.stores<<",\"group\":"<<d.group<<",\"req_k\":"<<d.req_k<<",\"issued\":"<<d.issued<<",\"acc0\":"<<signed32(d.acc[0])<<",\"acc1\":"<<signed32(d.acc[1])<<"}";d.debug=s.str();return d.debug.c_str();}
extern "C" const char *tlpu_device_last_error(){return error.c_str();}
extern "C" void tlpu_device_destroy(void*p){delete static_cast<Device*>(p);}
