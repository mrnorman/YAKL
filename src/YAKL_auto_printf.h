#pragma once
// Included by YAKL.h

namespace yakl {

  /** @private */
  inline void auto_printf( std::string const & operation , std::initializer_list<std::string> labels = {} ,
                           std::string const & metadata = "" ) {
    if constexpr (yakl_auto_printf) {
      auto & stream = get_yakl_instance().auto_printf_stream;
      stream << operation;
      bool first = true;
      for (auto const & label : labels) {
        if (label.empty()) continue;
        stream << (first ? ": " : ", ") << label;
        first = false;
      }
      if (!metadata.empty()) stream << " " << metadata;
      stream << std::endl;
      if (!stream) {
        std::cerr << "WARNING: YAKL_AUTO_PRINTF failed while writing: " << operation << std::endl;
        if (stream.bad())  std::cerr << "WARNING: YAKL_AUTO_PRINTF output stream has badbit set"  << std::endl;
        if (stream.fail()) std::cerr << "WARNING: YAKL_AUTO_PRINTF output stream has failbit set" << std::endl;
      }
    }
  }

  /** @private */
  inline void auto_begin_printf( std::string const & operation , std::initializer_list<std::string> labels = {} ,
                                 std::string const & metadata = "" ) {
    if constexpr (yakl_auto_printf) {
      Kokkos::fence();
      auto_printf("BEGIN "+operation,labels,metadata);
    }
  }

  /** @private */
  inline void auto_end_printf( std::string const & operation , std::initializer_list<std::string> labels = {} ,
                               std::string const & metadata = "" ) {
    if constexpr (yakl_auto_fence || yakl_auto_printf) Kokkos::fence();
    if constexpr (yakl_auto_printf) auto_printf("END "+operation,labels,metadata);
  }

  /** @private */
  inline void auto_fence_and_printf( std::string const & operation ,
                                     std::initializer_list<std::string> labels = {} ,
                                     std::string const & metadata = "" ) {
    if constexpr (yakl_auto_fence || yakl_auto_printf) Kokkos::fence();
    if constexpr (yakl_auto_printf) auto_printf(operation,labels,metadata);
  }

  /** @private */
  template <class View>
  inline std::string auto_array_metadata(View const & view) {
    std::ostringstream metadata;
    metadata << "rank=" << View::rank() << " extents=";
    for (int dim=0; dim < View::rank(); dim++) metadata << (dim == 0 ? "" : "x") << view.extent(dim);
    if constexpr (View::is_fstyle) {
      metadata << " bounds=";
      for (int dim=0; dim < View::rank(); dim++) {
        metadata << (dim == 0 ? "" : "x") << view.lb[dim] << ":";
        if (view.extent(dim) == 0) metadata << "empty";
        else metadata << view.lb[dim]+static_cast<index_t>(view.extent(dim))-1;
      }
    }
    metadata << " elements=" << view.size()
             << " bytes=" << view.size()*sizeof(typename View::non_const_value_type)
             << " space=" << (View::on_device ? "device" : "host");
    return metadata.str();
  }

} // namespace yakl
