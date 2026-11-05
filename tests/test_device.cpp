#include "tinylpu_device.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
constexpr unsigned enabled=(1u<<11)|(1u<<12);
std::vector<unsigned> run(const std::vector<unsigned>& bytes,unsigned lanes=2){
  void*d=tlpu_device_create(128,lanes);assert(d);
  tlpu_device_tick(d,0);tlpu_device_tick(d,enabled|(1u<<10));
  unsigned cursor=0;std::vector<unsigned> result;bool done=false;
  for(unsigned cycle=0;cycle<20000;++cycle){
    unsigned in=enabled;
    if(cursor<bytes.size()&&cycle%3!=0)in|=(1u<<8)|bytes[cursor];
    if(cycle%5!=0)in|=1u<<9;
    unsigned out=tlpu_device_tick(d,in),ev=tlpu_device_events(d);
    if(ev&1)++cursor;if(ev&2)result.push_back((ev>>8)&255);
    assert(!(out&(64u<<8)));
    if(out&(32u<<8)){done=true;break;}
  }
  assert(done&&cursor==bytes.size());tlpu_device_destroy(d);return result;
}
void transfers_and_wrap(){
  // Two rows: (-8)*(-8)=64; (-8)*7=-56. Runtime bytes, no golden backfill.
  std::vector<unsigned> p={1,0,0,0,0x08,1,0,8,1,0x08,0x07,
    3,0,0,8,32,1,0,1,2,0,32,7,15,0,0,0};
  for(unsigned lanes:{1u,2u})assert(run(p,lanes)==std::vector<unsigned>({64,0,0,0,200,255,255,255}));
  // Signed ACC32 wrap: INT32_MAX seed + 1 -> INT32_MIN.
  p={1,0,0,0,1,1,0,8,0,1,1,0,32,3,255,255,255,127,
     3,1,0,8,32,0,0,1,2,0,32,3,15,0,0,0};
  assert(run(p)==std::vector<unsigned>({0,0,0,128}));
}
void reset_error_and_start(){
  assert(!tlpu_device_create(31,2));assert(!tlpu_device_create(128,3));
  void*d=tlpu_device_create(128,2);assert(d);
  assert(tlpu_device_tick(d,0)==0);
  unsigned out=tlpu_device_tick(d,enabled|(1u<<10));assert(out&(4u<<8));
  // Start edge while a partial header is busy is an error; not retriggered
  // by holding the same high level on consecutive cycles.
  out=tlpu_device_tick(d,enabled|(1u<<10));assert(!(out&(64u<<8)));
  tlpu_device_tick(d,enabled);
  out=tlpu_device_tick(d,enabled|(1u<<10));assert(out&(64u<<8));
  out=tlpu_device_tick(d,1u<<11);assert(out==0); // ena abort
  out=tlpu_device_tick(d,enabled);assert(!(out&(16u<<8)));
  tlpu_device_tick(d,enabled|(1u<<10));
  for(unsigned b:{15u,0u,0u,0u})tlpu_device_tick(d,enabled|(1u<<8)|b);
  tlpu_device_tick(d,enabled);out=tlpu_device_tick(d,enabled);assert(out&(32u<<8));
  // Fresh start invalidates scratch; STORE must fault on uninitialized read.
  tlpu_device_tick(d,enabled|(1u<<10));
  for(unsigned b:{2u,0u,0u,0u})tlpu_device_tick(d,enabled|(1u<<8)|b);
  for(unsigned i=0;i<5;++i)out=tlpu_device_tick(d,enabled);
  assert(out&(64u<<8));tlpu_device_destroy(d);
}
}
int main(){transfers_and_wrap();reset_error_and_start();std::cout<<"device tests passed\n";}
