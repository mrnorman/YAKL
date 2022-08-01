
#pragma once

namespace yakl {
  
  class InitConfig {
  protected:
    std::function<void *( size_t , char const *)> alloc_host_func;
    std::function<void *( size_t , char const *)> alloc_device_func;
    std::function<void ( void * , char const *)>  free_host_func;
    std::function<void ( void * , char const *)>  free_device_func;  
    std::function<void ()>                        timer_init;
    std::function<void ()>                        timer_finalize;
    std::function<void (char const *)>            timer_start;
    std::function<void (char const *)>            timer_stop;
  
  public:
    InitConfig set_host_allocator  ( std::function<void *( size_t )> func ) {
      alloc_host_func   = [=] (size_t bytes , char const *label) -> void * { return func(bytes); };
      return *this;
    }
    InitConfig set_device_allocator( std::function<void *( size_t )> func ) {
      alloc_device_func = [=] (size_t bytes , char const *label) -> void * { return func(bytes); };
      return *this;
    }
    InitConfig set_host_deallocator  ( std::function<void ( void * )> func ) {
      free_host_func    = [=] (void *ptr , char const *label) { func(ptr); };
      return *this;
    }
    InitConfig set_device_deallocator( std::function<void ( void * )> func ) {
      free_device_func  = [=] (void *ptr , char const *label) { func(ptr); };
      return *this;
    }
    InitConfig set_host_allocator    ( std::function<void *( size_t , char const *)> func ) { alloc_host_func   = func; return *this; }
    InitConfig set_device_allocator  ( std::function<void *( size_t , char const *)> func ) { alloc_device_func = func; return *this; }
    InitConfig set_host_deallocator  ( std::function<void ( void * , char const *)>  func ) { free_host_func    = func; return *this; }
    InitConfig set_device_deallocator( std::function<void ( void * , char const *)>  func ) { free_device_func  = func; return *this; }
    InitConfig set_timer_init        ( std::function<void (            )>            func ) { timer_init      = func; return *this; }
    InitConfig set_timer_finalize    ( std::function<void (            )>            func ) { timer_finalize  = func; return *this; }
    InitConfig set_timer_start       ( std::function<void (char const *)>            func ) { timer_start     = func; return *this; }
    InitConfig set_timer_stop        ( std::function<void (char const *)>            func ) { timer_stop      = func; return *this; }
    std::function<void *( size_t , char const *)> get_host_allocator    () const { return alloc_host_func  ; }
    std::function<void *( size_t , char const *)> get_device_allocator  () const { return alloc_device_func; }
    std::function<void ( void * , char const *)>  get_host_deallocator  () const { return free_host_func   ; }
    std::function<void ( void * , char const *)>  get_device_deallocator() const { return free_device_func ; }
    std::function<void ()>                        get_timer_init        () const { return timer_init     ; }
    std::function<void ()>                        get_timer_finalize    () const { return timer_finalize ; }
    std::function<void (char const *)>            get_timer_start       () const { return timer_start    ; }
    std::function<void (char const *)>            get_timer_stop        () const { return timer_stop     ; }
  };

}


