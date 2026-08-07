program Fortran_Gator_Validation
  use iso_c_binding
  use gator_mod
#ifdef HAVE_MPI
  use mpi
#endif
  implicit none
  character(len=32) :: scenario
  integer(c_size_t) :: bytes
  integer, pointer :: arr(:) => null(), arr_slice(:) => null()
#ifdef HAVE_MPI
  integer :: ierr
  call MPI_Init(ierr)
#endif

  call get_command_argument(1,scenario)
  select case (trim(scenario))
    case ("positive")
      bytes = gator_checked_bytes((/1024,2048/),8_c_size_t)
      if (bytes /= 16777216_c_size_t) error stop "ERROR: safe byte-count calculation is incorrect"
      bytes = checked_bounds(bytes,(/2,3/),(/-huge(0),huge(0)-2/))
      if (bytes /= 16777216_c_size_t) error stop "ERROR: valid extreme lower bounds changed the byte count"
    case ("zero")
      bytes = gator_checked_bytes((/1024,0/),8_c_size_t)
    case ("negative")
      bytes = gator_checked_bytes((/1024,-1/),8_c_size_t)
    case ("overflow")
      bytes = gator_checked_bytes((/huge(0),huge(0)/),16_c_size_t)
    case ("associated_allocate")
      call gator_init()
      call gator_allocate(arr,(/4/))
      call gator_allocate(arr,(/8/))
    case ("disassociated_deallocate")
      call gator_init()
      call gator_deallocate(arr)
    case ("interior_deallocate")
      call gator_init()
      call gator_allocate(arr,(/4/))
      arr_slice => arr(2:)
      call gator_deallocate(arr_slice)
    case ("zero_extent")
      call gator_init()
      call gator_allocate(arr,(/0/))
    case ("lower_bound_overflow")
      call gator_allocate(arr,(/2/),(/huge(0)/))
    case default
      error stop "ERROR: unknown Fortran Gator validation scenario"
  end select
#ifdef HAVE_MPI
  call MPI_Finalize(ierr)
#endif
end program Fortran_Gator_Validation
