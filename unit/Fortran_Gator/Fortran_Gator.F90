
program Fortran_Gator
  use gator_mod
#ifdef HAVE_MPI
  use mpi
#endif
  implicit none
  real(8), pointer, contiguous :: a(:) => null(), b(:) => null(), c(:) => null(), d(:) => null()
#ifdef HAVE_MPI
  integer :: ierr
  call MPI_Init(ierr)
#endif
  call gator_init()
  call gator_allocate( a , (/1024*1024*70/) , (/-1/) )
  call gator_allocate( b , (/1024*1024*70/) , (/0/) )
  call gator_allocate( c , (/1024*1024*70/) , (/1/) )
  call gator_allocate( d , (/1024*1024*70/) , (/2/) )
  call gator_deallocate( a )
  call gator_deallocate( b )
  call gator_deallocate( c )
  call gator_deallocate( d )
  if (associated(a) .or. associated(b) .or. associated(c) .or. associated(d)) then
    error stop "ERROR: gator_deallocate did not disassociate a pointer"
  endif
  call gator_allocate( a , (/8/) , (/-4/) )
  if (lbound(a,1) /= -4 .or. ubound(a,1) /= 3) error stop "ERROR: gator_allocate returned incorrect bounds"
  call gator_deallocate( a )
  call gator_finalize()
#ifdef HAVE_MPI
  call MPI_Finalize(ierr)
#endif
end program Fortran_Gator
