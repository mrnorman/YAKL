

module gator_mod
  use iso_c_binding
  implicit none
  integer    :: i4
  integer(8) :: i8
  real       :: r4
  real(8)    :: r8
  logical    :: lg

  interface gator_init
    subroutine gator_init() bind(C, name="gatorInit")
    end subroutine gator_init
  end interface gator_init

  interface
    subroutine gator_finalize() bind(C, name="gatorFinalize")
    end subroutine gator_finalize

    function gator_allocate_c( bytes ) result(ptr) bind(C, name="gatorAllocate")
      use iso_c_binding
      type(c_ptr)              :: ptr
      integer(c_size_t), value :: bytes
    end function gator_allocate_c

    subroutine gator_deallocate_c( ptr ) bind(C, name="gatorDeallocate")
      use iso_c_binding
      type(c_ptr)      , value :: ptr
    end subroutine gator_deallocate_c
  end interface


  interface gator_allocate
    module procedure :: gator_allocate_int4_1d
    module procedure :: gator_allocate_int4_2d
    module procedure :: gator_allocate_int4_3d
    module procedure :: gator_allocate_int4_4d
    module procedure :: gator_allocate_int4_5d
    module procedure :: gator_allocate_int4_6d
    module procedure :: gator_allocate_int4_7d

    module procedure :: gator_allocate_int8_1d
    module procedure :: gator_allocate_int8_2d
    module procedure :: gator_allocate_int8_3d
    module procedure :: gator_allocate_int8_4d
    module procedure :: gator_allocate_int8_5d
    module procedure :: gator_allocate_int8_6d
    module procedure :: gator_allocate_int8_7d

    module procedure :: gator_allocate_real4_1d
    module procedure :: gator_allocate_real4_2d
    module procedure :: gator_allocate_real4_3d
    module procedure :: gator_allocate_real4_4d
    module procedure :: gator_allocate_real4_5d
    module procedure :: gator_allocate_real4_6d
    module procedure :: gator_allocate_real4_7d

    module procedure :: gator_allocate_real8_1d
    module procedure :: gator_allocate_real8_2d
    module procedure :: gator_allocate_real8_3d
    module procedure :: gator_allocate_real8_4d
    module procedure :: gator_allocate_real8_5d
    module procedure :: gator_allocate_real8_6d
    module procedure :: gator_allocate_real8_7d

    module procedure :: gator_allocate_log_1d
    module procedure :: gator_allocate_log_2d
    module procedure :: gator_allocate_log_3d
    module procedure :: gator_allocate_log_4d
    module procedure :: gator_allocate_log_5d
    module procedure :: gator_allocate_log_6d
    module procedure :: gator_allocate_log_7d
  end interface gator_allocate


  interface gator_deallocate
    module procedure :: gator_deallocate_int4_1d
    module procedure :: gator_deallocate_int4_2d
    module procedure :: gator_deallocate_int4_3d
    module procedure :: gator_deallocate_int4_4d
    module procedure :: gator_deallocate_int4_5d
    module procedure :: gator_deallocate_int4_6d
    module procedure :: gator_deallocate_int4_7d

    module procedure :: gator_deallocate_int8_1d
    module procedure :: gator_deallocate_int8_2d
    module procedure :: gator_deallocate_int8_3d
    module procedure :: gator_deallocate_int8_4d
    module procedure :: gator_deallocate_int8_5d
    module procedure :: gator_deallocate_int8_6d
    module procedure :: gator_deallocate_int8_7d

    module procedure :: gator_deallocate_real4_1d
    module procedure :: gator_deallocate_real4_2d
    module procedure :: gator_deallocate_real4_3d
    module procedure :: gator_deallocate_real4_4d
    module procedure :: gator_deallocate_real4_5d
    module procedure :: gator_deallocate_real4_6d
    module procedure :: gator_deallocate_real4_7d

    module procedure :: gator_deallocate_real8_1d
    module procedure :: gator_deallocate_real8_2d
    module procedure :: gator_deallocate_real8_3d
    module procedure :: gator_deallocate_real8_4d
    module procedure :: gator_deallocate_real8_5d
    module procedure :: gator_deallocate_real8_6d
    module procedure :: gator_deallocate_real8_7d

    module procedure :: gator_deallocate_log_1d
    module procedure :: gator_deallocate_log_2d
    module procedure :: gator_deallocate_log_3d
    module procedure :: gator_deallocate_log_4d
    module procedure :: gator_deallocate_log_5d
    module procedure :: gator_deallocate_log_6d
    module procedure :: gator_deallocate_log_7d
  end interface gator_deallocate


contains



  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! gator_allocate
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine gator_allocate_int4_1d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 1
    integer, pointer , intent(  out) :: arr       (:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):) => arr
  end subroutine gator_allocate_int4_1d
  subroutine gator_allocate_int4_2d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 2
    integer, pointer , intent(  out) :: arr       (:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):) => arr
  end subroutine gator_allocate_int4_2d
  subroutine gator_allocate_int4_3d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 3
    integer, pointer , intent(  out) :: arr       (:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):) => arr
  end subroutine gator_allocate_int4_3d
  subroutine gator_allocate_int4_4d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 4
    integer, pointer , intent(  out) :: arr       (:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => arr
  end subroutine gator_allocate_int4_4d
  subroutine gator_allocate_int4_5d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 5
    integer, pointer , intent(  out) :: arr       (:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => arr
  end subroutine gator_allocate_int4_5d
  subroutine gator_allocate_int4_6d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 6
    integer, pointer , intent(  out) :: arr       (:,:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => arr
  end subroutine gator_allocate_int4_6d
  subroutine gator_allocate_int4_7d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 7
    integer, pointer , intent(  out) :: arr       (:,:,:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => arr
  end subroutine gator_allocate_int4_7d

  subroutine gator_allocate_int8_1d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 1
    integer(8), pointer , intent(  out) :: arr       (:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):) => arr
  end subroutine gator_allocate_int8_1d
  subroutine gator_allocate_int8_2d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 2
    integer(8), pointer , intent(  out) :: arr       (:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):) => arr
  end subroutine gator_allocate_int8_2d
  subroutine gator_allocate_int8_3d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 3
    integer(8), pointer , intent(  out) :: arr       (:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):) => arr
  end subroutine gator_allocate_int8_3d
  subroutine gator_allocate_int8_4d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 4
    integer(8), pointer , intent(  out) :: arr       (:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => arr
  end subroutine gator_allocate_int8_4d
  subroutine gator_allocate_int8_5d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 5
    integer(8), pointer , intent(  out) :: arr       (:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => arr
  end subroutine gator_allocate_int8_5d
  subroutine gator_allocate_int8_6d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 6
    integer(8), pointer , intent(  out) :: arr       (:,:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => arr
  end subroutine gator_allocate_int8_6d
  subroutine gator_allocate_int8_7d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 7
    integer(8), pointer , intent(  out) :: arr       (:,:,:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(i8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => arr
  end subroutine gator_allocate_int8_7d

  subroutine gator_allocate_real4_1d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 1
    real   , pointer , intent(  out) :: arr       (:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):) => arr
  end subroutine gator_allocate_real4_1d
  subroutine gator_allocate_real4_2d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 2
    real   , pointer , intent(  out) :: arr       (:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):) => arr
  end subroutine gator_allocate_real4_2d
  subroutine gator_allocate_real4_3d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 3
    real   , pointer , intent(  out) :: arr       (:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):) => arr
  end subroutine gator_allocate_real4_3d
  subroutine gator_allocate_real4_4d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 4
    real   , pointer , intent(  out) :: arr       (:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => arr
  end subroutine gator_allocate_real4_4d
  subroutine gator_allocate_real4_5d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 5
    real   , pointer , intent(  out) :: arr       (:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => arr
  end subroutine gator_allocate_real4_5d
  subroutine gator_allocate_real4_6d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 6
    real   , pointer , intent(  out) :: arr       (:,:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => arr
  end subroutine gator_allocate_real4_6d
  subroutine gator_allocate_real4_7d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 7
    real   , pointer , intent(  out) :: arr       (:,:,:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r4),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => arr
  end subroutine gator_allocate_real4_7d

  subroutine gator_allocate_real8_1d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 1
    real(8), pointer , intent(  out) :: arr       (:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):) => arr
  end subroutine gator_allocate_real8_1d
  subroutine gator_allocate_real8_2d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 2
    real(8), pointer , intent(  out) :: arr       (:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):) => arr
  end subroutine gator_allocate_real8_2d
  subroutine gator_allocate_real8_3d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 3
    real(8), pointer , intent(  out) :: arr       (:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):) => arr
  end subroutine gator_allocate_real8_3d
  subroutine gator_allocate_real8_4d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 4
    real(8), pointer , intent(  out) :: arr       (:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => arr
  end subroutine gator_allocate_real8_4d
  subroutine gator_allocate_real8_5d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 5
    real(8), pointer , intent(  out) :: arr       (:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => arr
  end subroutine gator_allocate_real8_5d
  subroutine gator_allocate_real8_6d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 6
    real(8), pointer , intent(  out) :: arr       (:,:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => arr
  end subroutine gator_allocate_real8_6d
  subroutine gator_allocate_real8_7d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 7
    real(8), pointer , intent(  out) :: arr       (:,:,:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(r8),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => arr
  end subroutine gator_allocate_real8_7d

  subroutine gator_allocate_log_1d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 1
    logical, pointer , intent(  out) :: arr       (:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(lg),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):) => arr
  end subroutine gator_allocate_log_1d
  subroutine gator_allocate_log_2d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 2
    logical, pointer , intent(  out) :: arr       (:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(lg),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):) => arr
  end subroutine gator_allocate_log_2d
  subroutine gator_allocate_log_3d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 3
    logical, pointer , intent(  out) :: arr       (:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(lg),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):) => arr
  end subroutine gator_allocate_log_3d
  subroutine gator_allocate_log_4d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 4
    logical, pointer , intent(  out) :: arr       (:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(lg),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => arr
  end subroutine gator_allocate_log_4d
  subroutine gator_allocate_log_5d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 5
    logical, pointer , intent(  out) :: arr       (:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(lg),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => arr
  end subroutine gator_allocate_log_5d
  subroutine gator_allocate_log_6d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 6
    logical, pointer , intent(  out) :: arr       (:,:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(lg),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => arr
  end subroutine gator_allocate_log_6d
  subroutine gator_allocate_log_7d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 7
    logical, pointer , intent(  out) :: arr       (:,:,:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( int(product(dims)*sizeof(lg),c_size_t) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => arr
  end subroutine gator_allocate_log_7d



  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! gator_deallocate
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine gator_deallocate_int4_1d( arr )
    integer, pointer, intent(inout) :: arr(:)
    call gator_deallocate_c( c_loc( arr ) )
    arr=> NULL()
  end subroutine gator_deallocate_int4_1d
  subroutine gator_deallocate_int4_2d( arr )
    integer, pointer, intent(inout) :: arr(:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int4_2d
  subroutine gator_deallocate_int4_3d( arr )
    integer, pointer, intent(inout) :: arr(:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int4_3d
  subroutine gator_deallocate_int4_4d( arr )
    integer, pointer, intent(inout) :: arr(:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int4_4d
  subroutine gator_deallocate_int4_5d( arr )
    integer, pointer, intent(inout) :: arr(:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int4_5d
  subroutine gator_deallocate_int4_6d( arr )
    integer, pointer, intent(inout) :: arr(:,:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int4_6d
  subroutine gator_deallocate_int4_7d( arr )
    integer, pointer, intent(inout) :: arr(:,:,:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int4_7d

  subroutine gator_deallocate_int8_1d( arr )
    integer(8), pointer, intent(inout) :: arr(:)
    call gator_deallocate_c( c_loc( arr ) )
    arr=> NULL()
  end subroutine gator_deallocate_int8_1d
  subroutine gator_deallocate_int8_2d( arr )
    integer(8), pointer, intent(inout) :: arr(:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int8_2d
  subroutine gator_deallocate_int8_3d( arr )
    integer(8), pointer, intent(inout) :: arr(:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int8_3d
  subroutine gator_deallocate_int8_4d( arr )
    integer(8), pointer, intent(inout) :: arr(:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int8_4d
  subroutine gator_deallocate_int8_5d( arr )
    integer(8), pointer, intent(inout) :: arr(:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int8_5d
  subroutine gator_deallocate_int8_6d( arr )
    integer(8), pointer, intent(inout) :: arr(:,:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int8_6d
  subroutine gator_deallocate_int8_7d( arr )
    integer(8), pointer, intent(inout) :: arr(:,:,:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int8_7d

  subroutine gator_deallocate_real4_1d( arr )
    real, pointer, intent(inout) :: arr(:)
    call gator_deallocate_c( c_loc( arr ) )
    arr=> NULL()
  end subroutine gator_deallocate_real4_1d
  subroutine gator_deallocate_real4_2d( arr )
    real, pointer, intent(inout) :: arr(:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real4_2d
  subroutine gator_deallocate_real4_3d( arr )
    real, pointer, intent(inout) :: arr(:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real4_3d
  subroutine gator_deallocate_real4_4d( arr )
    real, pointer, intent(inout) :: arr(:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real4_4d
  subroutine gator_deallocate_real4_5d( arr )
    real, pointer, intent(inout) :: arr(:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real4_5d
  subroutine gator_deallocate_real4_6d( arr )
    real, pointer, intent(inout) :: arr(:,:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real4_6d
  subroutine gator_deallocate_real4_7d( arr )
    real, pointer, intent(inout) :: arr(:,:,:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real4_7d

  subroutine gator_deallocate_real8_1d( arr )
    real(8), pointer, intent(inout) :: arr(:)
    call gator_deallocate_c( c_loc( arr ) )
    arr=> NULL()
  end subroutine gator_deallocate_real8_1d
  subroutine gator_deallocate_real8_2d( arr )
    real(8), pointer, intent(inout) :: arr(:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real8_2d
  subroutine gator_deallocate_real8_3d( arr )
    real(8), pointer, intent(inout) :: arr(:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real8_3d
  subroutine gator_deallocate_real8_4d( arr )
    real(8), pointer, intent(inout) :: arr(:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real8_4d
  subroutine gator_deallocate_real8_5d( arr )
    real(8), pointer, intent(inout) :: arr(:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real8_5d
  subroutine gator_deallocate_real8_6d( arr )
    real(8), pointer, intent(inout) :: arr(:,:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real8_6d
  subroutine gator_deallocate_real8_7d( arr )
    real(8), pointer, intent(inout) :: arr(:,:,:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real8_7d

  subroutine gator_deallocate_log_1d( arr )
    logical, pointer, intent(inout) :: arr(:)
    call gator_deallocate_c( c_loc( arr ) )
    arr=> NULL()
  end subroutine gator_deallocate_log_1d
  subroutine gator_deallocate_log_2d( arr )
    logical, pointer, intent(inout) :: arr(:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_log_2d
  subroutine gator_deallocate_log_3d( arr )
    logical, pointer, intent(inout) :: arr(:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_log_3d
  subroutine gator_deallocate_log_4d( arr )
    logical, pointer, intent(inout) :: arr(:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_log_4d
  subroutine gator_deallocate_log_5d( arr )
    logical, pointer, intent(inout) :: arr(:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_log_5d
  subroutine gator_deallocate_log_6d( arr )
    logical, pointer, intent(inout) :: arr(:,:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_log_6d
  subroutine gator_deallocate_log_7d( arr )
    logical, pointer, intent(inout) :: arr(:,:,:,:,:,:,:)
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_log_7d


end module gator_mod

