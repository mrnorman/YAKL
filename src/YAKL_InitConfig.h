
#pragma once

#include <unistd.h>

namespace yakl {

  class InitConfig {
  protected:
    bool pool_enabled;
    size_t pool_size_mb;
    size_t pool_block_bytes;

  public:
    InitConfig() : pool_enabled(false) , pool_size_mb(0) , pool_block_bytes(256) { }
    InitConfig set_pool_enabled    ( bool enabled      ) { this->pool_enabled     = enabled    ; return *this; }
    InitConfig set_pool_size_mb    ( size_t size_mb    ) { this->pool_size_mb     = size_mb    ; return *this; }
    InitConfig set_pool_block_bytes( size_t block_bytes) { this->pool_block_bytes = block_bytes; return *this; }
    bool   get_pool_enabled    () const { return pool_enabled    ; }
    size_t get_pool_size_mb    () const { return pool_size_mb    ; }
    size_t get_pool_block_bytes() const { return pool_block_bytes; }
  };

}
