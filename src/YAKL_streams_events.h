
#pragma once

namespace yakl {

  #ifdef YAKL_ENABLE_STREAMS
    bool constexpr streams_enabled = true;
  #else
    bool constexpr streams_enabled = false;
  #endif

  #if   defined(YAKL_ARCH_CUDA)

    class Stream;
    class Event;

    class Stream {
      protected:
      cudaStream_t my_stream;
      int *        refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free

      void nullify() { my_stream = 0; refCount = nullptr; }

      public:

      Stream() { nullify(); }
      Stream(cudaStream_t cuda_stream) { nullify(); my_stream = cuda_stream; }
      ~Stream() { destroy(); }

      Stream(Stream const  &rhs) {
        my_stream = rhs.my_stream;
        refCount  = rhs.refCount;
        if (refCount != nullptr) (*refCount)++;
      }
      Stream(Stream &&rhs) {
        my_stream = rhs.my_stream;
        refCount  = rhs.refCount;
        rhs.nullify();
      }
      Stream & operator=(Stream const  &rhs) {
        if (this != &rhs) {
          destroy();
          my_stream = rhs.my_stream;
          refCount  = rhs.refCount;
          if (refCount != nullptr) (*refCount)++;
        }
        return *this;
      }
      Stream & operator=(Stream &&rhs) {
        if (this != &rhs) {
          destroy();
          my_stream = rhs.my_stream;
          refCount  = rhs.refCount;
          rhs.nullify();
        }
        return *this;
      }

      void create() {
        if (refCount == nullptr) {
          refCount = new int;
          (*refCount) = 1;
          if constexpr (streams_enabled) cudaStreamCreate( &my_stream );
        }
      }

      void destroy() {
        if (refCount != nullptr) {
          (*refCount)--;
          if ( (*refCount) == 0 ) {
            if constexpr (streams_enabled) cudaStreamDestroy( my_stream );
            delete refCount;
            nullify();
          }
        }
      }

      cudaStream_t get_real_stream() { return my_stream; }
      bool operator==(Stream stream) const { return my_stream == stream.get_real_stream(); }
      inline void wait_on_event(Event event);
      bool is_default_stream() { return my_stream == 0; }
      void fence() { cudaStreamSynchronize(my_stream); }
    };


    class Event {
      protected:
      cudaEvent_t my_event;
      int *       refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free

      void nullify() { my_event = 0; refCount = nullptr; }

      public:

      Event() { nullify(); }
      ~Event() { destroy(); }

      Event(Event const  &rhs) {
        my_event = rhs.my_event;
        refCount = rhs.refCount;
        if (refCount != nullptr) (*refCount)++;
      }
      Event(Event &&rhs) {
        my_event = rhs.my_event;
        refCount = rhs.refCount;
        rhs.nullify();
      }
      Event & operator=(Event const  &rhs) {
        if (this != &rhs) {
          destroy();
          my_event = rhs.my_event;
          refCount = rhs.refCount;
          if (refCount != nullptr) (*refCount)++;
        }
        return *this;
      }
      Event & operator=(Event &&rhs) {
        if (this != &rhs) {
          destroy();
          my_event = rhs.my_event;
          refCount = rhs.refCount;
          rhs.nullify();
        }
        return *this;
      }

      void create() {
        if (refCount == nullptr) {
          refCount = new int;
          (*refCount) = 1;
          cudaEventCreate( &my_event );
        }
      }

      void destroy() {
        if (refCount != nullptr) {
          (*refCount)--;
          if ( (*refCount) == 0 ) { cudaEventDestroy( my_event ); delete refCount; nullify(); }
        }
      }

      inline void record(Stream stream);
      cudaEvent_t get_real_event() { return my_event; }
      bool operator==(Event event) const { return my_event == event.get_real_event(); }
      bool completed() { return cudaEventQuery( my_event ) == cudaSuccess; }
      void fence() { cudaEventSynchronize(my_event); }
    };


    inline void Event::record(Stream stream) {
      create();
      cudaEventRecord( my_event , stream.get_real_stream() );
    }


    inline void Stream::wait_on_event(Event event) {
      cudaStreamWaitEvent( my_stream , event.get_real_event() );
    }

  #elif defined(YAKL_ARCH_HIP)

    class Stream;
    class Event;

    class Stream {
      protected:
      hipStream_t my_stream;
      int *        refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free

      void nullify() { my_stream = 0; refCount = nullptr; }

      public:

      Stream() { nullify(); }
      Stream(hipStream_t hip_stream) { nullify(); my_stream = hip_stream; }
      ~Stream() { destroy(); }

      Stream(Stream const  &rhs) {
        my_stream = rhs.my_stream;
        refCount  = rhs.refCount;
        if (refCount != nullptr) (*refCount)++;
      }
      Stream(Stream &&rhs) {
        my_stream = rhs.my_stream;
        refCount  = rhs.refCount;
        rhs.nullify();
      }
      Stream & operator=(Stream const  &rhs) {
        if (this != &rhs) {
          destroy();
          my_stream = rhs.my_stream;
          refCount  = rhs.refCount;
          if (refCount != nullptr) (*refCount)++;
        }
        return *this;
      }
      Stream & operator=(Stream &&rhs) {
        if (this != &rhs) {
          destroy();
          my_stream = rhs.my_stream;
          refCount  = rhs.refCount;
          rhs.nullify();
        }
        return *this;
      }

      void create() {
        if (refCount == nullptr) {
          refCount = new int;
          (*refCount) = 1;
          if constexpr (streams_enabled) hipStreamCreate( &my_stream );
        }
      }

      void destroy() {
        if (refCount != nullptr) {
          (*refCount)--;
          if ( (*refCount) == 0 ) {
            if constexpr (streams_enabled) hipStreamDestroy( my_stream );
            delete refCount;
            nullify();
          }
        }
      }

      hipStream_t get_real_stream() { return my_stream; }
      bool operator==(Stream stream) const { return my_stream == stream.get_real_stream(); }
      inline void wait_on_event(Event event);
      bool is_default_stream() { return my_stream == 0; }
      void fence() { hipStreamSynchronize(my_stream); }
    };


    class Event {
      protected:
      hipEvent_t my_event;
      int *       refCount;       // Pointer shared by multiple copies of this Array to keep track of allcation / free

      void nullify() { my_event = 0; refCount = nullptr; }

      public:

      Event() { nullify(); }
      ~Event() { destroy(); }

      Event(Event const  &rhs) {
        my_event = rhs.my_event;
        refCount = rhs.refCount;
        if (refCount != nullptr) (*refCount)++;
      }
      Event(Event &&rhs) {
        my_event = rhs.my_event;
        refCount = rhs.refCount;
        rhs.nullify();
      }
      Event & operator=(Event const  &rhs) {
        if (this != &rhs) {
          destroy();
          my_event = rhs.my_event;
          refCount = rhs.refCount;
          if (refCount != nullptr) (*refCount)++;
        }
        return *this;
      }
      Event & operator=(Event &&rhs) {
        if (this != &rhs) {
          destroy();
          my_event = rhs.my_event;
          refCount = rhs.refCount;
          rhs.nullify();
        }
        return *this;
      }

      void create() {
        if (refCount == nullptr) {
          refCount = new int;
          (*refCount) = 1;
          hipEventCreate( &my_event );
        }
      }

      void destroy() {
        if (refCount != nullptr) {
          (*refCount)--;
          if ( (*refCount) == 0 ) { hipEventDestroy( my_event ); delete refCount; nullify(); }
        }
      }

      inline void record(Stream stream);
      hipEvent_t get_real_event() { return my_event; }
      bool operator==(Event event) const { return my_event == event.get_real_event(); }
      bool completed() { return hipEventQuery( my_event ) == hipSuccess; }
      void fence() { hipEventSynchronize(my_event); }
    };


    inline void Event::record(Stream stream) {
      create();
      hipEventRecord( my_event , stream.get_real_stream() );
    }


    inline void Stream::wait_on_event(Event event) {
      hipStreamWaitEvent( my_stream , event.get_real_event() , 0 );
    }

  #elif defined(YAKL_ARCH_SYCL)

    struct Stream;
    struct Event;

    struct Stream {
      void create() { }
      void destroy() { }
      bool operator==(Stream stream) const { return true; }
      inline void wait_on_event(Event event);
      bool is_default_stream() { return true; }
      void fence() { }
    };

    struct Event {
      void create() { }
      void destroy() { }
      inline void record(Stream stream);
      bool operator==(Event event) const { return true; }
      bool completed() { return true; }
      void fence() { }
    };

    inline void Event::record(Stream stream) { }
    inline void Stream::wait_on_event(Event event) {  }

  #else

    struct Stream;
    struct Event;

    struct Stream {
      void create() { }
      void destroy() { }
      bool operator==(Stream stream) const { return true; }
      inline void wait_on_event(Event event);
      bool is_default_stream() { return true; }
      void fence() { }
    };

    struct Event {
      void create() { }
      void destroy() { }
      inline void record(Stream stream);
      bool operator==(Event event) const { return true; }
      bool completed() { return true; }
      void fence() { }
    };

    inline void Event::record(Stream stream) { }
    inline void Stream::wait_on_event(Event event) {  }

  #endif
    

  /** @brief Create and return a Stream object. It is guaranteed to not be the default stream */
  inline Stream create_stream() { Stream stream; stream.create(); return stream; }

  /** @brief Create, record, and return an event using the given stream */
  inline Event record_event(Stream stream = Stream()) { Event event; event.create(); event.record(stream); return event; }


  /** @brief Implements a list of Stream objects. 
    *        Needs to store a pointer to avoid construction on the device since Array objects need to store a
    *        list of streams on which they depend. */
  struct StreamList {
    std::vector<Stream> *list;
    YAKL_INLINE StreamList() {
      #if YAKL_CURRENTLY_ON_HOST()
        list = new std::vector<Stream>;
      #endif
    }
    YAKL_INLINE ~StreamList() {
      #if YAKL_CURRENTLY_ON_HOST()
        delete list;
      #endif
    }
    void push_back(Stream stream) {
      yakl_mtx_lock();
      list->push_back(stream);
      yakl_mtx_unlock();
    }
    int size() const { return list->size(); }
    bool empty() const { return list->empty(); }
    Stream operator[] (int i) { return (*list)[i]; }
    std::vector<Stream> get_all_streams() const { return *list; }
  };

}


