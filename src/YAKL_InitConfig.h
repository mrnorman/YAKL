
#pragma once

#include <unistd.h>

namespace yakl {

  class InitConfig {
  public:
    enum class PoolSetting { Default, Enabled, Disabled };

  protected:
    PoolSetting pool_setting;
    size_t      pool_size_mb;
    std::optional<size_t> pool_block_bytes;

  public:
    InitConfig() : pool_setting(PoolSetting::Default) , pool_size_mb(0) , pool_block_bytes(std::nullopt) { }
    InitConfig set_pool_enabled    ( bool enabled      ) {
      this->pool_setting = enabled ? PoolSetting::Enabled : PoolSetting::Disabled;
      return *this;
    }
    InitConfig set_pool_size_mb    ( size_t size_mb    ) { this->pool_size_mb     = size_mb    ; return *this; }
    InitConfig set_pool_block_bytes( size_t block_bytes) {
      if constexpr (kokkos_debug) {
        if (block_bytes == 0 || block_bytes%LinearAllocator::requiredAlignment != 0) {
          Kokkos::abort("ERROR: pool block size must be a positive multiple of Kokkos memory alignment");
        }
      }
      this->pool_block_bytes = block_bytes;
      return *this;
    }
    PoolSetting get_pool_setting    () const { return pool_setting    ; }
    bool        get_pool_enabled    () const { return pool_setting == PoolSetting::Enabled; }
    size_t      get_pool_size_mb    () const { return pool_size_mb    ; }
    size_t      get_pool_block_bytes() const { return pool_block_bytes.value_or(4096); }
    bool        pool_block_bytes_was_set() const { return pool_block_bytes.has_value(); }
  };

}
