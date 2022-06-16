
#pragma once

namespace yakl {
  
  class InitConfig {
  protected:
    std::function<void *( size_t , char const *)> yaklAllocHost;
    std::function<void *( size_t , char const *)> yaklAllocDevice;
    std::function<void ( void * , char const *)>  yaklFreeHost;
    std::function<void ( void * , char const *)>  yaklFreeDevice;  
    std::function<void ()>                        timer_init;
    std::function<void ()>                        timer_finalize;
    std::function<void (char const *)>            timer_start;
    std::function<void (char const *)>            timer_stop;
  
  public:
    InitConfig set_host_allocator  ( std::function<void *( size_t )> func ) {
      yaklAllocHost   = [=] (size_t bytes , char const *label) -> void * { return func(bytes); };
      return *this;
    }
    InitConfig set_device_allocator( std::function<void *( size_t )> func ) {
      yaklAllocDevice = [=] (size_t bytes , char const *label) -> void * { return func(bytes); };
      return *this;
    }
    InitConfig set_host_deallocator  ( std::function<void ( void * )> func ) {
      yaklFreeHost    = [=] (void *ptr , char const *label) { func(ptr); };
      return *this;
    }
    InitConfig set_device_deallocator( std::function<void ( void * )> func ) {
      yaklFreeDevice  = [=] (void *ptr , char const *label) { func(ptr); };
      return *this;
    }
    InitConfig set_host_allocator    ( std::function<void *( size_t , char const *)> func ) { yaklAllocHost   = func; return *this; }
    InitConfig set_device_allocator  ( std::function<void *( size_t , char const *)> func ) { yaklAllocDevice = func; return *this; }
    InitConfig set_host_deallocator  ( std::function<void ( void * , char const *)>  func ) { yaklFreeHost    = func; return *this; }
    InitConfig set_device_deallocator( std::function<void ( void * , char const *)>  func ) { yaklFreeDevice  = func; return *this; }
    InitConfig set_timer_init        ( std::function<void (            )>            func ) { timer_init      = func; return *this; }
    InitConfig set_timer_finalize    ( std::function<void (            )>            func ) { timer_finalize  = func; return *this; }
    InitConfig set_timer_start       ( std::function<void (char const *)>            func ) { timer_start     = func; return *this; }
    InitConfig set_timer_stop        ( std::function<void (char const *)>            func ) { timer_stop      = func; return *this; }
    std::function<void *( size_t , char const *)> get_host_allocator    () const { return yaklAllocHost  ; }
    std::function<void *( size_t , char const *)> get_device_allocator  () const { return yaklAllocDevice; }
    std::function<void ( void * , char const *)>  get_host_deallocator  () const { return yaklFreeHost   ; }
    std::function<void ( void * , char const *)>  get_device_deallocator() const { return yaklFreeDevice ; }
    std::function<void ()>                        get_timer_init        () const { return timer_init     ; }
    std::function<void ()>                        get_timer_finalize    () const { return timer_finalize ; }
    std::function<void (char const *)>            get_timer_start       () const { return timer_start    ; }
    std::function<void (char const *)>            get_timer_stop        () const { return timer_stop     ; }
  };

}


