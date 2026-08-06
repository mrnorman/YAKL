program Fortran_Gator_Validation
  use iso_c_binding
  use gator_mod, only: gator_checked_bytes
  implicit none
  character(len=16) :: scenario
  integer(c_size_t) :: bytes

  call get_command_argument(1,scenario)
  select case (trim(scenario))
    case ("positive")
      bytes = gator_checked_bytes((/1024,2048/),8_c_size_t)
      if (bytes /= 16777216_c_size_t) error stop "ERROR: safe byte-count calculation is incorrect"
    case ("zero")
      bytes = gator_checked_bytes((/1024,0/),8_c_size_t)
    case ("negative")
      bytes = gator_checked_bytes((/1024,-1/),8_c_size_t)
    case ("overflow")
      bytes = gator_checked_bytes((/huge(0),huge(0)/),16_c_size_t)
    case default
      error stop "ERROR: unknown Fortran Gator validation scenario"
  end select
end program Fortran_Gator_Validation
