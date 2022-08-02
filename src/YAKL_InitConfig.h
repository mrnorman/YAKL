/**
 * @file
 * An object of this class can optionally be passed to yakl::init() to configure the initialization
 */

#pragma once

namespace yakl {
  
  /** @brief An object of this class can optionally be passed to yakl::init() to configure the initialization.
    * @details This allows the user to override timer, allocation, and deallocation routines.
    * 
    * All `set_` functions return the InitConfig object they were called on. Therefore, the user can code, e.g.,
    * `yakl::init(yakl::InitConfig().set_device_allocator(myalloc).set_device_deallocator(myfree));` */
  class InitConfig {
  protected:
    /** @private */
    std::function<void *( size_t , char const *)> alloc_host_func;
    /** @private */
    std::function<void *( size_t , char const *)> alloc_device_func;
    /** @private */
    std::function<void ( void * , char const *)>  free_host_func;
    /** @private */
    std::function<void ( void * , char const *)>  free_device_func;  
    /** @private */
    std::function<void ()>                        timer_init;
    /** @private */
    std::function<void ()>                        timer_finalize;
    /** @private */
    std::function<void (char const *)>            timer_start;
    /** @private */
    std::function<void (char const *)>            timer_stop;
  
  public:
    /** @brief Pass the host allocator function you wish to use to override YAKL's default (NO LABEL) */
    InitConfig set_host_allocator  ( std::function<void *( size_t )> func ) {
      alloc_host_func   = [=] (size_t bytes , char const *label) -> void * { return func(bytes); };
      return *this;
    }
    /** @brief Pass the device allocator function you wish to use to override YAKL's default (NO LABEL) */
    InitConfig set_device_allocator( std::function<void *( size_t )> func ) {
      alloc_device_func = [=] (size_t bytes , char const *label) -> void * { return func(bytes); };
      return *this;
    }
    /** @brief Pass the host deallocator function you wish to use to override YAKL's default (NO LABEL) */
    InitConfig set_host_deallocator  ( std::function<void ( void * )> func ) {
      free_host_func    = [=] (void *ptr , char const *label) { func(ptr); };
      return *this;
    }
    /** @brief Pass the device deallocator function you wish to use to override YAKL's default (NO LABEL) */
    InitConfig set_device_deallocator( std::function<void ( void * )> func ) {
      free_device_func  = [=] (void *ptr , char const *label) { func(ptr); };
      return *this;
    }
    /** @brief Pass the host allocator function you wish to use to override YAKL's default (LABEL) */
    InitConfig set_host_allocator    ( std::function<void *( size_t , char const *)> func ) { alloc_host_func   = func; return *this; }
    /** @brief Pass the device allocator function you wish to use to override YAKL's default (LABEL) */
    InitConfig set_device_allocator  ( std::function<void *( size_t , char const *)> func ) { alloc_device_func = func; return *this; }
    /** @brief Pass the host deallocator function you wish to use to override YAKL's default (LABEL) */
    InitConfig set_host_deallocator  ( std::function<void ( void * , char const *)>  func ) { free_host_func    = func; return *this; }
    /** @brief Pass the device deallocator function you wish to use to override YAKL's default (LABEL) */
    InitConfig set_device_deallocator( std::function<void ( void * , char const *)>  func ) { free_device_func  = func; return *this; }
    /** @brief Pass the timer init function you wish to use to override YAKL's default */
    InitConfig set_timer_init        ( std::function<void (            )>            func ) { timer_init      = func; return *this; }
    /** @brief Pass the timer finalize function you wish to use to override YAKL's default */
    InitConfig set_timer_finalize    ( std::function<void (            )>            func ) { timer_finalize  = func; return *this; }
    /** @brief Pass the timer start function you wish to use to override YAKL's default */
    InitConfig set_timer_start       ( std::function<void (char const *)>            func ) { timer_start     = func; return *this; }
    /** @brief Pass the timer stop function you wish to use to override YAKL's default */
    InitConfig set_timer_stop        ( std::function<void (char const *)>            func ) { timer_stop      = func; return *this; }
    /** @private */
    std::function<void *( size_t , char const *)> get_host_allocator    () const { return alloc_host_func  ; }
    /** @private */
    std::function<void *( size_t , char const *)> get_device_allocator  () const { return alloc_device_func; }
    /** @private */
    std::function<void ( void * , char const *)>  get_host_deallocator  () const { return free_host_func   ; }
    /** @private */
    std::function<void ( void * , char const *)>  get_device_deallocator() const { return free_device_func ; }
    /** @private */
    std::function<void ()>                        get_timer_init        () const { return timer_init     ; }
    /** @private */
    std::function<void ()>                        get_timer_finalize    () const { return timer_finalize ; }
    /** @private */
    std::function<void (char const *)>            get_timer_start       () const { return timer_start    ; }
    /** @private */
    std::function<void (char const *)>            get_timer_stop        () const { return timer_stop     ; }
  };

}


