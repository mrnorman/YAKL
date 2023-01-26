
#pragma once

/** @brief The ArrayIR namespace holds the ArrayIR class and memory type constants associated with ArrayIR.
           This class holds library-agnostic Array metadata to make it easy to transfer array objects between
           different C++ libraries. */
namespace array_ir {

  /** @brief Declares that the data pointer is defined only in host memory */
  int constexpr MEMORY_HOST   = 0;
  /** @brief Declares that the data pointer is defined only in device memory */
  int constexpr MEMORY_DEVICE = 1;
  /** @brief Declares that the data pointer is defined in both host and device memory */
  int constexpr MEMORY_SHARED = 2;


  /** @brief The ArrayIR class holds library-agnostic Array metadata to make it easy to transfer array objects between
   *         different C++ libraries 
   *
   * This holds intermediate representation of Array information for transformation between C++ portability libraries
   * ArrayIR is only valid on the host and cannot be used directly in kernels. Please transform to your portability
   *   library's Array representation first, and then use that inside kernels.
   * ArrayIR pointers are always assumed to point to contiguous data
   * Dimensions passed to the constructor and accessed via extent() are always assumed to have the right-most
   *   dimension varying the fastest.
   * You purposefully cannot use an operator() to get the data because you should only be using this class
   *   to transfer Array metadata from one C++ portability framework to another.
   * The ArrayIR is completely un-owned, and it is up to the user to ensure data described by an ArrayIR
   *   object remains valid while being used by another portability framework.
   *
   * @param T The type of the array
   * @param N The rank (dimensionality) of the array
   */
  template <class T, int N>
  class ArrayIR {
    private:
    T                    * my_data;         // Data pointer
    std::array<size_t,N>   my_dimensions;   // Dimensions (right-most varying the fastest
    char const           * my_label;        // Label for the array (for debugging and things)
    int                    my_memory_type;  // Memory type (host, device, or shared)

    void nullify() { my_data = nullptr; my_label = nullptr; my_dimensions.fill(0); }

    void copy_from(ArrayIR const &rhs) {
      my_data        = rhs.my_data;
      my_label       = rhs.my_label;
      my_memory_type = rhs.my_memory_type;
      my_dimensions  = rhs.my_dimensions;
    }

    void error_message(char const *message) const {
      std::cerr << "*** ArrayIR class: A FATAL ERROR WAS ENCOUNTERED ***" << std::endl;
      std::cerr << message << std::endl;
      throw message;
    }

    
    public:
    /** @brief The exact type of this ArrayIR object with all qualifiers */
    typedef T                                 exact_type;
    /** @brief The type of this ArrayIR object with cv qualifiers removed */
    typedef typename std::remove_cv<T>::type  remove_cv_type;

    /** @brief Creates an empty object with nullptr as the data pointer */
    ArrayIR() { nullify(); }
    /** @brief Creates an ArrayIR object with the given data, dimensions, memory type, and optional label
      * 
      * @param  data         The data pointer
      * @param  dimensions   Dimensions with right-most varying the fastest. Initializer lists are allowed
      * @param  memory_type  array_ir::MEMORY_HOST if the data is valid only on the host.
      *                      array_ir::MEMORY_DEVICE if the data is valid only on the device.
      *                      array_ir::MEMORY_SHARED if the data is valid on both host and device.
      * @param  label        Optional label parameter for debugging and such things. */
    ArrayIR( T * data , std::array<size_t,N> dimensions , int memory_type , char const * label="" ) {
      my_data        = data;
      my_memory_type = memory_type;
      my_label       = label;
      my_dimensions  = dimensions;
    }
    /** @brief Copy constructor */
    ArrayIR           (ArrayIR const  &rhs) { copy_from(rhs); }
    /** @brief Move constructor */
    ArrayIR           (ArrayIR const &&rhs) { copy_from(rhs); }
    /** @brief Copy constructor */
    ArrayIR &operator=(ArrayIR const  &rhs) { copy_from(rhs); }
    /** @brief Move constructor */
    ArrayIR &operator=(ArrayIR const &&rhs) { copy_from(rhs); }
    ~ArrayIR() { nullify(); }
    /** @brief Is this ArrayIR object invalid (does it point to nullptr)? */
    bool empty   () const { return my_data == nullptr; }
    /** @brief Is this ArrayIR object invalid (does it point to nullptr)? */
    bool is_empty() const { return my_data == nullptr; }
    /** @brief Is this ArrayIR object valid (does it point to something other than nullptr)? */
    bool valid   () const { return ! empty(); }
    /** @brief Is this ArrayIR object valid (does it point to something other than nullptr)? */
    bool is_valid() const { return ! empty(); }
    /** @brief Get the data pointer */
    T * data            () const { return my_data; }
    /** @brief Get the data pointer */
    T * get_data        () const { return my_data; }
    /** @brief Get the data pointer */
    T * get_data_pointer() const { return my_data; }
    /** @brief Get the extent of the dimension of the provided index */
    size_t extent(int i) const {
      if (i < 0 || i > N-1) error_message("extent() index out of bounds");
      return my_dimensions[i];
    }
    /** @brief Get the shape of the array as a std::array<size_t,N> */
    std::array<size_t,N> shape() const { return my_dimensions; }
    /** @brief Get the extent of the dimension of the provided index */
    size_t dimension(int i) const { return extent(i); }
    /** @brief Get the label for this array */
    char const * label    () const { return my_label; }
    /** @brief Get the label for this array */
    char const * get_label() const { return my_label; }
    /** @brief Get the memory type for this array object (MEMORY_HOST, MEMORY_DEVICE, or MEMORY_SHARED) */
    int memory_type    () const { return my_memory_type; }
    /** @brief Get the memory type for this array object (MEMORY_HOST, MEMORY_DEVICE, or MEMORY_SHARED) */
    int get_memory_type() const { return my_memory_type; }
    /** @brief Determine if the data pointer is valid on the host */
    bool data_valid_on_host  () const { return (my_memory_type == MEMORY_HOST  ) || (my_memory_type == MEMORY_SHARED); }
    /** @brief Determine if the data pointer is valid on the device */
    bool data_valid_on_device() const { return (my_memory_type == MEMORY_DEVICE) || (my_memory_type == MEMORY_SHARED); }
  };

}


