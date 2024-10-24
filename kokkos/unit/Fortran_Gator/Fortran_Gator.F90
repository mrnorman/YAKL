
program Fortran_Gator
  use gator_mod
  implicit none
  real(8), pointer, contiguous :: a(:), b(:), c(:), d(:)
  call gator_init()
  call gator_allocate( a , (/1024*1024*70/) , (/-1/) )
  call gator_allocate( b , (/1024*1024*70/) , (/0/) )
  call gator_allocate( c , (/1024*1024*70/) , (/1/) )
  call gator_allocate( d , (/1024*1024*70/) , (/2/) )
  call gator_deallocate( a )
  call gator_deallocate( b )
  call gator_deallocate( c )
  call gator_deallocate( d )
  call gator_finalize()
end program Fortran_Gator
