
#pragma once

namespace yakl {
  struct CStyle { static constexpr bool is_cstyle = true; using layout = Kokkos::LayoutRight; };
  struct FStyle { static constexpr bool is_fstyle = true; using layout = Kokkos::LayoutLeft ; };

  template <class Type> inline constexpr bool is_CStyle = requires { Type::is_cstyle; };
  template <class Type> inline constexpr bool is_FStyle = requires { Type::is_fstyle; };
}

