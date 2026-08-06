

! These are Fortran routines to initialize YAKL, finalize YAKL, and allocate and deallocate
! through YAKL's memory pool so that the same data is accessible in both C++ and Fortran codes
! Also gives Fortran access to efficient managed memory allocations through YAKL's pool

module gator_mod
  use iso_c_binding
  implicit none
  integer    :: i4
  integer(8) :: i8
  real       :: r4
  real(8)    :: r8
  complex    :: c4
  complex(8) :: c8
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

    module procedure :: gator_allocate_cplx4_1d
    module procedure :: gator_allocate_cplx4_2d
    module procedure :: gator_allocate_cplx4_3d
    module procedure :: gator_allocate_cplx4_4d
    module procedure :: gator_allocate_cplx4_5d
    module procedure :: gator_allocate_cplx4_6d
    module procedure :: gator_allocate_cplx4_7d

    module procedure :: gator_allocate_cplx8_1d
    module procedure :: gator_allocate_cplx8_2d
    module procedure :: gator_allocate_cplx8_3d
    module procedure :: gator_allocate_cplx8_4d
    module procedure :: gator_allocate_cplx8_5d
    module procedure :: gator_allocate_cplx8_6d
    module procedure :: gator_allocate_cplx8_7d

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

    module procedure :: gator_deallocate_cplx4_1d
    module procedure :: gator_deallocate_cplx4_2d
    module procedure :: gator_deallocate_cplx4_3d
    module procedure :: gator_deallocate_cplx4_4d
    module procedure :: gator_deallocate_cplx4_5d
    module procedure :: gator_deallocate_cplx4_6d
    module procedure :: gator_deallocate_cplx4_7d

    module procedure :: gator_deallocate_cplx8_1d
    module procedure :: gator_deallocate_cplx8_2d
    module procedure :: gator_deallocate_cplx8_3d
    module procedure :: gator_deallocate_cplx8_4d
    module procedure :: gator_deallocate_cplx8_5d
    module procedure :: gator_deallocate_cplx8_6d
    module procedure :: gator_deallocate_cplx8_7d

    module procedure :: gator_deallocate_log_1d
    module procedure :: gator_deallocate_log_2d
    module procedure :: gator_deallocate_log_3d
    module procedure :: gator_deallocate_log_4d
    module procedure :: gator_deallocate_log_5d
    module procedure :: gator_deallocate_log_6d
    module procedure :: gator_deallocate_log_7d
  end interface gator_deallocate


contains


  function gator_checked_bytes(dims,element_bytes) result(num_bytes)
    use iso_c_binding
    integer, intent(in) :: dims(:)
    integer(c_size_t), value :: element_bytes
    integer(c_size_t) :: num_bytes
    integer :: i
    num_bytes = 1_c_size_t
    if (element_bytes == 0_c_size_t) error stop "ERROR: gator_allocate element size must be positive"
    do i = 1, size(dims)
      if (dims(i) <= 0) error stop "ERROR: gator_allocate dimensions must be positive"
      if (int(dims(i),c_size_t) > huge(num_bytes)/num_bytes) then
        error stop "ERROR: gator_allocate dimension product overflow"
      endif
      num_bytes = num_bytes*int(dims(i),c_size_t)
    enddo
    if (num_bytes > huge(num_bytes)/element_bytes) error stop "ERROR: gator_allocate byte-count overflow"
    num_bytes = num_bytes*element_bytes
  end function gator_checked_bytes


  function gator_checked_allocate(bytes,already_associated) result(ptr)
    use iso_c_binding
    integer(c_size_t), value :: bytes
    logical, value :: already_associated
    type(c_ptr) :: ptr
    if (already_associated) error stop "ERROR: gator_allocate called with an associated pointer"
    ptr = gator_allocate_c(bytes)
    if (.not. c_associated(ptr)) error stop "ERROR: gator_allocate returned a null C pointer"
  end function gator_checked_allocate


  subroutine gator_checked_deallocate(ptr)
    use iso_c_binding
    type(c_ptr), value :: ptr
    if (.not. c_associated(ptr)) error stop "ERROR: gator_deallocate received a null C pointer"
    call gator_deallocate_c(ptr)
  end subroutine gator_checked_deallocate


#define out inout
#define gator_allocate_c(bytes) gator_checked_allocate(bytes,associated(arr))
#define gator_deallocate_c(ptr) gator_checked_deallocate(ptr)



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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(i8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r4),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r8),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(r8),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => arr
  end subroutine gator_allocate_real8_7d

  subroutine gator_allocate_cplx4_1d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 1
    complex, pointer , intent(  out) :: arr       (:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c4),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):) => arr
  end subroutine gator_allocate_cplx4_1d
  subroutine gator_allocate_cplx4_2d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 2
    complex, pointer , intent(  out) :: arr       (:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c4),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):) => arr
  end subroutine gator_allocate_cplx4_2d
  subroutine gator_allocate_cplx4_3d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 3
    complex, pointer , intent(  out) :: arr       (:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c4),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):) => arr
  end subroutine gator_allocate_cplx4_3d
  subroutine gator_allocate_cplx4_4d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 4
    complex, pointer , intent(  out) :: arr       (:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c4),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => arr
  end subroutine gator_allocate_cplx4_4d
  subroutine gator_allocate_cplx4_5d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 5
    complex, pointer , intent(  out) :: arr       (:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c4),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => arr
  end subroutine gator_allocate_cplx4_5d
  subroutine gator_allocate_cplx4_6d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 6
    complex, pointer , intent(  out) :: arr       (:,:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c4),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => arr
  end subroutine gator_allocate_cplx4_6d
  subroutine gator_allocate_cplx4_7d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 7
    complex, pointer , intent(  out) :: arr       (:,:,:,:,:,:,:)
    integer          , intent(in   ) :: dims      (ndims)
    integer, optional, intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c4),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => arr
  end subroutine gator_allocate_cplx4_7d

  subroutine gator_allocate_cplx8_1d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 1
    complex(8), pointer , intent(  out) :: arr       (:)
    integer             , intent(in   ) :: dims      (ndims)
    integer, optional   , intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c8),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):) => arr
  end subroutine gator_allocate_cplx8_1d
  subroutine gator_allocate_cplx8_2d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 2
    complex(8), pointer , intent(  out) :: arr       (:,:)
    integer             , intent(in   ) :: dims      (ndims)
    integer, optional   , intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c8),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):) => arr
  end subroutine gator_allocate_cplx8_2d
  subroutine gator_allocate_cplx8_3d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 3
    complex(8), pointer , intent(  out) :: arr       (:,:,:)
    integer             , intent(in   ) :: dims      (ndims)
    integer, optional   , intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c8),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):) => arr
  end subroutine gator_allocate_cplx8_3d
  subroutine gator_allocate_cplx8_4d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 4
    complex(8), pointer , intent(  out) :: arr       (:,:,:,:)
    integer             , intent(in   ) :: dims      (ndims)
    integer, optional   , intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c8),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):) => arr
  end subroutine gator_allocate_cplx8_4d
  subroutine gator_allocate_cplx8_5d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 5
    complex(8), pointer , intent(  out) :: arr       (:,:,:,:,:)
    integer             , intent(in   ) :: dims      (ndims)
    integer, optional   , intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c8),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):) => arr
  end subroutine gator_allocate_cplx8_5d
  subroutine gator_allocate_cplx8_6d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 6
    complex(8), pointer , intent(  out) :: arr       (:,:,:,:,:,:)
    integer             , intent(in   ) :: dims      (ndims)
    integer, optional   , intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c8),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):) => arr
  end subroutine gator_allocate_cplx8_6d
  subroutine gator_allocate_cplx8_7d( arr , dims , lbounds_in )
    integer, parameter :: ndims = 7
    complex(8), pointer , intent(  out) :: arr       (:,:,:,:,:,:,:)
    integer             , intent(in   ) :: dims      (ndims)
    integer, optional   , intent(in   ) :: lbounds_in(ndims)
    integer :: lbounds(ndims)
    type(c_ptr) :: data_ptr
    if (present(lbounds_in)) then
      lbounds = lbounds_in
    else
      lbounds = 1
    endif
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(c8),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => arr
  end subroutine gator_allocate_cplx8_7d

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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(lg),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(lg),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(lg),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(lg),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(lg),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(lg),c_size_t)) )
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
    data_ptr = gator_allocate_c( gator_checked_bytes(dims,int(sizeof(lg),c_size_t)) )
    call c_f_pointer( data_ptr , arr , dims )
    arr(lbounds(1):,lbounds(2):,lbounds(3):,lbounds(4):,lbounds(5):,lbounds(6):,lbounds(7):) => arr
  end subroutine gator_allocate_log_7d



  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! gator_deallocate
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine gator_deallocate_int4_1d( arr )
    integer, pointer, intent(inout) :: arr(:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr=> NULL()
  end subroutine gator_deallocate_int4_1d
  subroutine gator_deallocate_int4_2d( arr )
    integer, pointer, intent(inout) :: arr(:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int4_2d
  subroutine gator_deallocate_int4_3d( arr )
    integer, pointer, intent(inout) :: arr(:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int4_3d
  subroutine gator_deallocate_int4_4d( arr )
    integer, pointer, intent(inout) :: arr(:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int4_4d
  subroutine gator_deallocate_int4_5d( arr )
    integer, pointer, intent(inout) :: arr(:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int4_5d
  subroutine gator_deallocate_int4_6d( arr )
    integer, pointer, intent(inout) :: arr(:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int4_6d
  subroutine gator_deallocate_int4_7d( arr )
    integer, pointer, intent(inout) :: arr(:,:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int4_7d

  subroutine gator_deallocate_int8_1d( arr )
    integer(8), pointer, intent(inout) :: arr(:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr=> NULL()
  end subroutine gator_deallocate_int8_1d
  subroutine gator_deallocate_int8_2d( arr )
    integer(8), pointer, intent(inout) :: arr(:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int8_2d
  subroutine gator_deallocate_int8_3d( arr )
    integer(8), pointer, intent(inout) :: arr(:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int8_3d
  subroutine gator_deallocate_int8_4d( arr )
    integer(8), pointer, intent(inout) :: arr(:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int8_4d
  subroutine gator_deallocate_int8_5d( arr )
    integer(8), pointer, intent(inout) :: arr(:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int8_5d
  subroutine gator_deallocate_int8_6d( arr )
    integer(8), pointer, intent(inout) :: arr(:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int8_6d
  subroutine gator_deallocate_int8_7d( arr )
    integer(8), pointer, intent(inout) :: arr(:,:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_int8_7d

  subroutine gator_deallocate_real4_1d( arr )
    real, pointer, intent(inout) :: arr(:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr=> NULL()
  end subroutine gator_deallocate_real4_1d
  subroutine gator_deallocate_real4_2d( arr )
    real, pointer, intent(inout) :: arr(:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real4_2d
  subroutine gator_deallocate_real4_3d( arr )
    real, pointer, intent(inout) :: arr(:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real4_3d
  subroutine gator_deallocate_real4_4d( arr )
    real, pointer, intent(inout) :: arr(:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real4_4d
  subroutine gator_deallocate_real4_5d( arr )
    real, pointer, intent(inout) :: arr(:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real4_5d
  subroutine gator_deallocate_real4_6d( arr )
    real, pointer, intent(inout) :: arr(:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real4_6d
  subroutine gator_deallocate_real4_7d( arr )
    real, pointer, intent(inout) :: arr(:,:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real4_7d

  subroutine gator_deallocate_real8_1d( arr )
    real(8), pointer, intent(inout) :: arr(:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr=> NULL()
  end subroutine gator_deallocate_real8_1d
  subroutine gator_deallocate_real8_2d( arr )
    real(8), pointer, intent(inout) :: arr(:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real8_2d
  subroutine gator_deallocate_real8_3d( arr )
    real(8), pointer, intent(inout) :: arr(:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real8_3d
  subroutine gator_deallocate_real8_4d( arr )
    real(8), pointer, intent(inout) :: arr(:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real8_4d
  subroutine gator_deallocate_real8_5d( arr )
    real(8), pointer, intent(inout) :: arr(:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real8_5d
  subroutine gator_deallocate_real8_6d( arr )
    real(8), pointer, intent(inout) :: arr(:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real8_6d
  subroutine gator_deallocate_real8_7d( arr )
    real(8), pointer, intent(inout) :: arr(:,:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_real8_7d

  subroutine gator_deallocate_cplx4_1d( arr )
    complex, pointer, intent(inout) :: arr(:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr=> NULL()
  end subroutine gator_deallocate_cplx4_1d
  subroutine gator_deallocate_cplx4_2d( arr )
    complex, pointer, intent(inout) :: arr(:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_cplx4_2d
  subroutine gator_deallocate_cplx4_3d( arr )
    complex, pointer, intent(inout) :: arr(:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_cplx4_3d
  subroutine gator_deallocate_cplx4_4d( arr )
    complex, pointer, intent(inout) :: arr(:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_cplx4_4d
  subroutine gator_deallocate_cplx4_5d( arr )
    complex, pointer, intent(inout) :: arr(:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_cplx4_5d
  subroutine gator_deallocate_cplx4_6d( arr )
    complex, pointer, intent(inout) :: arr(:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_cplx4_6d
  subroutine gator_deallocate_cplx4_7d( arr )
    complex, pointer, intent(inout) :: arr(:,:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_cplx4_7d

  subroutine gator_deallocate_cplx8_1d( arr )
    complex(8), pointer, intent(inout) :: arr(:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr=> NULL()
  end subroutine gator_deallocate_cplx8_1d
  subroutine gator_deallocate_cplx8_2d( arr )
    complex(8), pointer, intent(inout) :: arr(:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_cplx8_2d
  subroutine gator_deallocate_cplx8_3d( arr )
    complex(8), pointer, intent(inout) :: arr(:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_cplx8_3d
  subroutine gator_deallocate_cplx8_4d( arr )
    complex(8), pointer, intent(inout) :: arr(:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_cplx8_4d
  subroutine gator_deallocate_cplx8_5d( arr )
    complex(8), pointer, intent(inout) :: arr(:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_cplx8_5d
  subroutine gator_deallocate_cplx8_6d( arr )
    complex(8), pointer, intent(inout) :: arr(:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_cplx8_6d
  subroutine gator_deallocate_cplx8_7d( arr )
    complex(8), pointer, intent(inout) :: arr(:,:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_cplx8_7d

  subroutine gator_deallocate_log_1d( arr )
    logical, pointer, intent(inout) :: arr(:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr=> NULL()
  end subroutine gator_deallocate_log_1d
  subroutine gator_deallocate_log_2d( arr )
    logical, pointer, intent(inout) :: arr(:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_log_2d
  subroutine gator_deallocate_log_3d( arr )
    logical, pointer, intent(inout) :: arr(:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_log_3d
  subroutine gator_deallocate_log_4d( arr )
    logical, pointer, intent(inout) :: arr(:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_log_4d
  subroutine gator_deallocate_log_5d( arr )
    logical, pointer, intent(inout) :: arr(:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_log_5d
  subroutine gator_deallocate_log_6d( arr )
    logical, pointer, intent(inout) :: arr(:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_log_6d
  subroutine gator_deallocate_log_7d( arr )
    logical, pointer, intent(inout) :: arr(:,:,:,:,:,:,:)
    if (.not. associated(arr)) error stop "ERROR: gator_deallocate called with a disassociated pointer"
    call gator_deallocate_c( c_loc( arr ) )
    arr => NULL()
  end subroutine gator_deallocate_log_7d


#undef gator_deallocate_c
#undef gator_allocate_c
#undef out

end module gator_mod
