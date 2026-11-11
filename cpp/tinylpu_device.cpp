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
  unsigned slot=0; bool last=false;
};
// One product packet carries the low and high nibble of a fetched byte.
struct Product {bool valid=false,last=false; unsigned count=0; std::array<int,2> value{};};
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
  unsigned header_index=0,byte_index=0,group=0,active=0,seed_lane=0;
  unsigned req_byte=0,issued=0; bool fetch_done=false;
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
    // STORE reads one byte ahead of the TX register, and a second while TX waits.
    bool store=state==StoreRequest||(state==StoreResponse&&byte_index!=h[3])||
               (state==Transmit&&byte_index+2<=h[3]);
    // A 4B-aligned seed occupies all four banks; the next lane's read overlaps.
    bool seed=state==SeedRequest||(state==SeedResponse&&seed_lane+1!=active);
    if(store) {
      unsigned address=h[2]+byte_index+(state==StoreRequest?0u:state==StoreResponse?1u:2u);
      r.enable[address%4]=true;r.address[address%4]=address;
    } else if(seed) {
      unsigned base=h[4]+(group+seed_lane+(state==SeedResponse?1u:0u))*4;
      for(unsigned b=0;b<4;++b){r.enable[b]=true;r.address[b]=base+b;}
    } else if(state==Core&&!fetch_done) {
      std::array<unsigned,3> addresses={h[2]+req_byte,h[3]+group*h[7]+req_byte,h[3]+(group+1)*h[7]+req_byte};
      unsigned needed=active==2?7u:3u;
      for(unsigned role=0;role<3;++role) {
        if(!(needed&(1u<<role))||(issued&(1u<<role)))continue;
        unsigned bank=addresses[role]%4;
        if(!r.enable[bank]){r.enable[bank]=true;r.address[bank]=addresses[role];}
        if(r.address[bank]==addresses[role])r.roles[bank]|=1u<<role;
      }
      r.slot=req_byte%2;r.last=req_byte==h[6]/2;
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
    state=Idle;previous_start=false;h={};header_index=byte_index=group=active=seed_lane=0;
    req_byte=issued=0;fetch_done=false;a_end=w_end=y_read_end=y_write_end=tx=0;
    std::fill(valid.begin(),valid.end(),false);q={};qvalid={};pending={};buffer={};mask={};acc={};pipe={};
    commands=macs=loads=stores=0;mvm_bad=false;
  }
  std::uint32_t write_value(unsigned lane) const {
    std::int64_t value=signed32(acc[lane]);
    if(out_bytes()==2)value=std::max<std::int64_t>(-32768,std::min<std::int64_t>(32767,value));
    return static_cast<std::uint32_t>(value);
  }
  unsigned tick(unsigned in) {
    ++cycles;event=0;
    unsigned before=output(in);
    bool rxvalid=in&(1u<<8),ready=in&(1u<<9),edge=start_edge(in);
    bool accepted_rx=(before&(4u<<8))&&rxvalid;
    bool accepted_tx=(before&(8u<<8))&&ready;
    Request r=request();
    bool clear=false;unsigned write=0,address=0;std::uint32_t data=0;
    if(!(in&(1u<<11))||!(in&(1u<<12))) {
      reset();clear=true;r={};
    } else if(edge) {
      previous_start=true;r={};pending={};pipe={};
      if(state==Idle||state==Done){reset();previous_start=true;state=Header;clear=true;}
      else state=Error;
    } else {
      previous_start=(in&(1u<<10))!=0;
      bool bad=(state==Check&&bad_header())||(state==GroupInit&&mvm_bad);
      if(state==StoreResponse)bad=bad||!qvalid[(h[2]+byte_index)%4];
      if(state==SeedResponse)for(unsigned b=0;b<4;++b)bad=bad||!qvalid[b];
      if(state==Core)for(unsigned b=0;b<4;++b)if(pending.roles[b]&&!qvalid[b])bad=true;
      if(bad){state=Error;r={};pending={};pipe={};}
      else switch(state) {
        case Idle:case Done:case Error:break;
        case Header:
          if(accepted_rx){
            event|=1;
            if(header_index==0)h={};
            h[header_index]=in&255;
            if(header_index==(h[0]==3?7u:3u)){state=h[0]==3?Ends:Check;header_index=0;}
            else ++header_index;
          }break;
        case Ends:
          a_end=h[2]+row_bytes();w_end=h[3]+h[5]*h[7]+row_bytes();
          y_read_end=h[4]+((h[1]&1u)?rows()*4:0);y_write_end=h[4]+rows()*out_bytes();
          state=MvmValidate;break;
        case MvmValidate:
          mvm_bad=bad_header();byte_index=0;group=0;state=GroupInit;break;
        case Check:
          ++commands;byte_index=0;
          if(h[0]==1)state=Load;
          if(h[0]==2)state=StoreRequest;
          if(h[0]==15)state=Done;
          break;
        case Load:
          if(accepted_rx){event|=1;write=1;address=h[2]+byte_index;data=in&255;++loads;
            if(byte_index==h[3]){state=Header;header_index=0;}else ++byte_index;
          }break;
        case StoreRequest:state=StoreResponse;break;
        case StoreResponse:tx=q[(h[2]+byte_index)%4];state=Transmit;break;
        case Transmit:
          if(accepted_tx){event|=2|(tx<<8);++stores;
            if(byte_index==h[3]){state=Header;header_index=0;}
            else if(!qvalid[(h[2]+byte_index+1)%4]){state=Error;pending={};pipe={};}
            else{++byte_index;tx=q[(h[2]+byte_index)%4];}
          }break;
        case GroupInit:
          if(group==0)++commands;
          active=std::min(lanes,rows()-group);acc={};pipe={};mask={};buffer={};
          req_byte=issued=seed_lane=byte_index=0;fetch_done=false;
          state=(h[1]&1u)?SeedRequest:Core;break;
        case SeedRequest:state=SeedResponse;break;
        case SeedResponse:{
          acc[seed_lane]=q[0]|(q[1]<<8)|(q[2]<<16)|(q[3]<<24);
          if(seed_lane+1==active)state=Core;else ++seed_lane;
          break;}
        case Core:{
          Product commit=pipe[1];pipe[1]=pipe[0];pipe[0]={};
          if(commit.valid){for(unsigned lane=0;lane<active;++lane)acc[lane]+=static_cast<std::uint32_t>(commit.value[lane]);macs+=active*commit.count;
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
              // Odd K ignores the high nibble of the last byte.
              bool both=!(pending.last&&h[6]%2==0);
              pipe[0].valid=true;pipe[0].last=pending.last;pipe[0].count=both?2:1;
              for(unsigned lane=0;lane<active;++lane)
                pipe[0].value[lane]=nibble(buffer[slot][0],0)*nibble(buffer[slot][lane+1],0)+
                  (both?nibble(buffer[slot][0],1)*nibble(buffer[slot][lane+1],1):0);
            }
          }
          if(!fetch_done){
            if(issued==0)mask[r.slot]=0;
            unsigned combined=issued;for(unsigned b=0;b<4;++b)combined|=r.roles[b];
            if(combined==(active==2?7u:3u)){issued=0;if(req_byte==h[6]/2)fetch_done=true;else ++req_byte;}
            else issued=combined;
          }break;}
        case Writeback:
          // One lane per clock; byte_index is the lane.
          write=out_bytes();address=h[4]+(group+byte_index)*out_bytes();data=write_value(byte_index);
          if(byte_index+1==active){
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
      for(unsigned i=0;i<write;++i){
        if(address+i>=capacity)throw std::logic_error("unchecked write address");
        memory[address+i]=(data>>(8*i))&255;valid[address+i]=true;
      }
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
extern "C" const char *tlpu_device_debug(void*p){if(!p)return "{}";auto&d=*static_cast<Device*>(p);std::ostringstream s;s<<"{\"state\":"<<d.state<<",\"cycles\":"<<d.cycles<<",\"commands\":"<<d.commands<<",\"macs\":"<<d.macs<<",\"load_bytes\":"<<d.loads<<",\"store_bytes\":"<<d.stores<<",\"group\":"<<d.group<<",\"req_byte\":"<<d.req_byte<<",\"issued\":"<<d.issued<<",\"acc0\":"<<signed32(d.acc[0])<<",\"acc1\":"<<signed32(d.acc[1])<<"}";d.debug=s.str();return d.debug.c_str();}
extern "C" const char *tlpu_device_last_error(){return error.c_str();}
extern "C" void tlpu_device_destroy(void*p){delete static_cast<Device*>(p);}
