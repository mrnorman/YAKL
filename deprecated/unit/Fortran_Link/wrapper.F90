
program wrapper
  use gator_mod
  implicit none
  integer, parameter :: n = 1024
  real :: a(n), b(n), c(n)

  interface init
    subroutine add(a, b, c, n) bind(C,name="add")
      use iso_c_binding
      real, dimension(*) :: a, b, c
      integer :: n
    end subroutine
  end interface

  call gator_init()

  call add(a,b,c,n)

  if (sum(a) / n /= 3 .or. sum(b) / n /= 1 .or. sum(c) / n /= 2 ) then
    write(*,*) 'ERROR: wrong sum'
    stop
  endif

  call gator_finalize()

end program wrapper
