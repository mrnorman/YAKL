
#pragma once
// IMPORTANT: THIS FILE IS FOR DOCUMENTATION ONLY AND IS NOT USED!

namespace yakl {
  namespace c {

    // The functions are declared below so that I can document them in doxygen inside the appropriate namespace.
    /**
     * @brief [ASYNCHRONOUS] Launch the passed functor in parallel.
     * 
     * If passing a lambda, it must be decorated with YAKL_LAMBDA.
     * If passing a functor, the operator() must be decorated with YAKL_INLINE. Click for more information.
     * 
     * @param str    String label for this `parallel_for`. This form of `parallel_for` is highly recommended so that
     *               debugging and profiling features can be used when turned on via CPP macros.
     * @param bounds The yakl::c::Bounds or yakl::c::SimpleBounds object describing the tightly nested looping.
     *               You can also pass asingle integer, `{lower,upper}` pair, or `{lower,upper,stride}` triplet
     *               ensuring strides are positive. Why a positive stride? To protect you. If you **need** a negative
     *               stride, this means loop ordering matters, and the loop body is not a trivially parallel
     *               operation. The order of the bounds is always such that the left-most is the outer-most loop, and
     *               the right-most is the inner-most loop. All loop bounds expressed as a single integer, `N` will
     *               default to a lower bound of `0` and an upper bound of `N-1`. All loop bounds specified as 
     *               `{lower,upper}` or `{lower,upper,stride}` are **inclusive**, meaning the indices will be
     *               `{lower,lower+stride,...,upper_with_stride}`, where `upper_with_stride <= upper` depending on
     *               whether the index `upper` is exactly reached with the striding.
     * @param f      The functor to be launched in parallel. Lambdas must be decorated with YAKL_LAMBDA. Functors
     *               must have operator() decorated with YAKL_INLINE.
     * @param config [Optional] Use a yakl::LaunchConfig object to describe the size of inner-level parallelism
     *                          (`VecLen`) or whether this kernel should be executed in serial (`B4B`) when the CPP
     *                          macro `-DYAKL_B4B` is defined.
     */
    template <class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN , bool B4B = false>
    inline void parallel_for( char const * str , Bounds<N,simple> const &bounds , F const &f ,
                              LaunchConfig<VecLen,B4B> config = LaunchConfig<>() );

    /**
     * @brief [ASYNCHRONOUS] Launch the passed functor in parallel. 
     * 
     * Same as the other form of yakl::c::parallel_for but without the string label.
     */
    template <class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN , bool B4B = false>
    inline void parallel_for( Bounds<N,simple> const &bounds , F const &f ,
                              LaunchConfig<VecLen,B4B> config = LaunchConfig<>() );

    /**
     * @brief [ASYNCHRONOUS] Launch the passed functor in parallel in the coarsest-level parallelism on the device.
     * 
     * For hierarchical (two-level) parallelism only.
     * For CUDA and HIP,
     * for instance, this is "grid"-level parallelism spread over multiprocessors. yakl::c::parallel_inner, 
     * on the other hand, is "block"-level parallelism spread over threads within a multiprocessor.
     * If passing a lambda, it must be decorated with YAKL_LAMBDA.
     * If passing a functor, the operator() must be decorated with YAKL_INLINE. Click for more information.
     * IMPORTANT: While the yakl::LaunchConfig parameter is optional, you will very likely want to use it!
     * Otherwise, you're at the mercy of the YAKL_DEFAULT_VECTOR_LEN for a given hardware backend.
     * The yakl::LaunchConfig parameter's template vector length parameter **must be larger than** the `inner_size`
     * declared by yakl::LaunchConfig::set_inner_size(). Click for more information.
     * 
     * Example usage:
     * ```
     * int constexpr MAX_INNER_SIZE = 256;
     * int inner_size = 96;
     * yakl::c::parallel_outer( Bounds<2>(nz,{0,ny}) , YAKL_LAMBDA (int k, int j, InnerHandler handler) {
     *   ...
     * } , LaunchConfig<MAX_INNER_SIZE>.set_inner_size(inner_size) );
     * ```
     * 
     * IMPORTANT: **All** code inside yakl::c::parallel_outer is run in parallel over both outer and inner parallelism.
     *            So code not inside yakl::c::parallel_inner will still execute for **all** inner threads but without any
     *            knowledge of inner parallelism indices. If you want to execute only for one inner thread, please use
     *            the yakl::single_inner routine.
     * 
     * @param str    String label for this `parallel_outer`. This form of `parallel_outer` is highly recommended so that
     *               debugging and profiling features can be used when turned on via CPP macros.
     * @param bounds The yakl::c::Bounds or yakl::c::SimpleBounds object describing the tightly nested looping.
     *               You can also pass asingle integer, `{lower,upper}` pair, or `{lower,upper,stride}` triplet
     *               ensuring strides are positive. Why a positive stride? To protect you. If you **need** a negative
     *               stride, this means loop ordering matters, and the loop body is not a trivially parallel
     *               operation. The order of the bounds is always such that the left-most is the outer-most loop, and
     *               the right-most is the inner-most loop. All loop bounds expressed as a single integer, `N` will
     *               default to a lower bound of `0` and an upper bound of `N-1`. All loop bounds specified as 
     *               `{lower,upper}` or `{lower,upper,stride}` are **inclusive**, meaning the indices will be
     *               `{lower,lower+stride,...,upper_with_stride}`, where `upper_with_stride <= upper` depending on
     *               whether the index `upper` is exactly reached with the striding.
     * @param f      The functor to be launched in parallel. Lambdas must be decorated with YAKL_LAMBDA. Functors
     *               must have operator() decorated with YAKL_INLINE. IMPORTANT: The lambda or operator() **must**
     *               accept an additional yakl::InnerHandler object after the loop indices.
     * @param config [Optional, but HIGHLY ENCOURAGED] Use the `VecLen` template parameter to define the
     *               **maximum size** of the inner looping. When creating the yakl::LaunchConfig object, use the
     *               `yakl::LaunchConfig::set_inner_size(int)` routine to set the actual size of the inner looping.
     *               Ensure `set_inner_size <= VecLen`. Also an optional `B4B` template parameter to tell YAKL to
     *               run this kernel in serial when `-DYAKL_B4B` is defined as a CPP macro.
     */
    template <class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN, bool B4B = false>
    inline void parallel_outer( char const * str , Bounds<N,simple> const &bounds , F const &f ,
                                LaunchConfig<VecLen,B4B> config = LaunchConfig<>() );

    /**
     * @brief [ASYNCHRONOUS] Launch the passed functor in parallel in the coarsest-level parallelism on the device.
     * 
     * Same as the other form of yakl::c::parallel_outer but without the string label.
     */
    template <class F, int N, bool simple, int VecLen=YAKL_DEFAULT_VECTOR_LEN, bool B4B = false>
    inline void parallel_outer( Bounds<N,simple> const &bounds , F const &f ,
                                LaunchConfig<VecLen,B4B> config = LaunchConfig<>() );

    /**
     * @brief Launch the passed functor in parallel in the finenst-level parallelism on the device.
     * 
     * For hierarchical (two-level) parallelism only. **Must be called from within a yakl::c::parallel_outer call.**
     * Launch the passed functor in parallel in the finenst-level parallelism on the device. For CUDA and HIP,
     * for instance, this is "block"-level parallelism spread over threads within a multiprocessor. 
     * **IMPORTANT: If passing a lambda, it must be decorated with `[&]` and not `YAKL_LAMBDA`.**
     * If passing a functor, the operator() must **not** be decorated with YAKL_INLINE. Click for more information.
     *
     * @param bounds The yakl::c::Bounds or yakl::c::SimpleBounds object describing the tightly nested looping.
     *                You can also pass asingle integer, `{lower,upper}` pair, or `{lower,upper,stride}` triplet
     *                ensuring strides are positive. Why a positive stride? To protect you. If you **need** a negative
     *                stride, this means loop ordering matters, and the loop body is not a trivially parallel
     *                operation. The order of the bounds is always such that the left-most is the outer-most loop, and
     *                the right-most is the inner-most loop. All loop bounds expressed as a single integer, `N` will
     *                default to a lower bound of `0` and an upper bound of `N-1`. All loop bounds specified as 
     *                `{lower,upper}` or `{lower,upper,stride}` are **inclusive**, meaning the indices will be
     *                `{lower,lower+stride,...,upper_with_stride}`, where `upper_with_stride <= upper` depending on
     *                whether the index `upper` is exactly reached with the striding.
     * @param f       The functor to be launched in parallel. Lambdas must be decorated with YAKL_LAMBDA. Functors
     *                must have operator() decorated with `[&]` and **not** YAKL_INLINE.
     * @param handler yakl::InnerHandler object created by yakl::c::parallel_outer.
     */
    template <class F, int N, bool simple>
    YAKL_INLINE void parallel_inner( Bounds<N,simple> const &bounds , F const &f , InnerHandler handler );

    /**
     * @brief Launch the passed functor to only use one of the inner threads (still parallel over outer threads)
     * 
     * For hierarchical (two-level) parallelism only. **Must be called from within a yakl::c::parallel_outer call.**
     * Most of the time, you will use yakl::fence_inner() before and after yakl::c::single_inner.
     *
     * @param f       The functor to be launched in parallel. Lambdas must be decorated with YAKL_LAMBDA. Functors
     *                must have operator() decorated with `[&]` and **not** YAKL_INLINE.
     * @param handler yakl::InnerHandler object created by yakl::c::parallel_outer.
     */
    template <class F>
    YAKL_INLINE void single_inner( F const &f , InnerHandler handler );

  }
}

