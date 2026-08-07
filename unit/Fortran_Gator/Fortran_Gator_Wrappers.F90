#define TEST_WRAPPER(arr,dims) call gator_allocate(arr,dims); call gator_deallocate(arr)

program Fortran_Gator_Wrappers
  use gator_mod
#ifdef HAVE_MPI
  use mpi
#endif
  implicit none

  integer   , pointer :: i4_1(:), i4_2(:,:), i4_3(:,:,:), i4_4(:,:,:,:), i4_5(:,:,:,:,:)
  integer   , pointer :: i4_6(:,:,:,:,:,:), i4_7(:,:,:,:,:,:,:)
  integer(8), pointer :: i8_1(:), i8_2(:,:), i8_3(:,:,:), i8_4(:,:,:,:), i8_5(:,:,:,:,:)
  integer(8), pointer :: i8_6(:,:,:,:,:,:), i8_7(:,:,:,:,:,:,:)
  real      , pointer :: r4_1(:), r4_2(:,:), r4_3(:,:,:), r4_4(:,:,:,:), r4_5(:,:,:,:,:)
  real      , pointer :: r4_6(:,:,:,:,:,:), r4_7(:,:,:,:,:,:,:)
  real(8)   , pointer :: r8_1(:), r8_2(:,:), r8_3(:,:,:), r8_4(:,:,:,:), r8_5(:,:,:,:,:)
  real(8)   , pointer :: r8_6(:,:,:,:,:,:), r8_7(:,:,:,:,:,:,:)
  complex   , pointer :: c4_1(:), c4_2(:,:), c4_3(:,:,:), c4_4(:,:,:,:), c4_5(:,:,:,:,:)
  complex   , pointer :: c4_6(:,:,:,:,:,:), c4_7(:,:,:,:,:,:,:)
  complex(8), pointer :: c8_1(:), c8_2(:,:), c8_3(:,:,:), c8_4(:,:,:,:), c8_5(:,:,:,:,:)
  complex(8), pointer :: c8_6(:,:,:,:,:,:), c8_7(:,:,:,:,:,:,:)
  logical   , pointer :: lg_1(:), lg_2(:,:), lg_3(:,:,:), lg_4(:,:,:,:), lg_5(:,:,:,:,:)
  logical   , pointer :: lg_6(:,:,:,:,:,:), lg_7(:,:,:,:,:,:,:)
#ifdef HAVE_MPI
  integer :: ierr
#endif

  nullify(i4_1,i4_2,i4_3,i4_4,i4_5,i4_6,i4_7)
  nullify(i8_1,i8_2,i8_3,i8_4,i8_5,i8_6,i8_7)
  nullify(r4_1,r4_2,r4_3,r4_4,r4_5,r4_6,r4_7)
  nullify(r8_1,r8_2,r8_3,r8_4,r8_5,r8_6,r8_7)
  nullify(c4_1,c4_2,c4_3,c4_4,c4_5,c4_6,c4_7)
  nullify(c8_1,c8_2,c8_3,c8_4,c8_5,c8_6,c8_7)
  nullify(lg_1,lg_2,lg_3,lg_4,lg_5,lg_6,lg_7)

#ifdef HAVE_MPI
  call MPI_Init(ierr)
#endif
  call gator_init()

  TEST_WRAPPER(i4_1,(/2/))
  TEST_WRAPPER(i4_2,(/2,1/))
  TEST_WRAPPER(i4_3,(/2,1,1/))
  TEST_WRAPPER(i4_4,(/2,1,1,1/))
  TEST_WRAPPER(i4_5,(/2,1,1,1,1/))
  TEST_WRAPPER(i4_6,(/2,1,1,1,1,1/))
  TEST_WRAPPER(i4_7,(/2,1,1,1,1,1,1/))
  TEST_WRAPPER(i8_1,(/2/))
  TEST_WRAPPER(i8_2,(/2,1/))
  TEST_WRAPPER(i8_3,(/2,1,1/))
  TEST_WRAPPER(i8_4,(/2,1,1,1/))
  TEST_WRAPPER(i8_5,(/2,1,1,1,1/))
  TEST_WRAPPER(i8_6,(/2,1,1,1,1,1/))
  TEST_WRAPPER(i8_7,(/2,1,1,1,1,1,1/))
  TEST_WRAPPER(r4_1,(/2/))
  TEST_WRAPPER(r4_2,(/2,1/))
  TEST_WRAPPER(r4_3,(/2,1,1/))
  TEST_WRAPPER(r4_4,(/2,1,1,1/))
  TEST_WRAPPER(r4_5,(/2,1,1,1,1/))
  TEST_WRAPPER(r4_6,(/2,1,1,1,1,1/))
  TEST_WRAPPER(r4_7,(/2,1,1,1,1,1,1/))
  TEST_WRAPPER(r8_1,(/2/))
  TEST_WRAPPER(r8_2,(/2,1/))
  TEST_WRAPPER(r8_3,(/2,1,1/))
  TEST_WRAPPER(r8_4,(/2,1,1,1/))
  TEST_WRAPPER(r8_5,(/2,1,1,1,1/))
  TEST_WRAPPER(r8_6,(/2,1,1,1,1,1/))
  TEST_WRAPPER(r8_7,(/2,1,1,1,1,1,1/))
  TEST_WRAPPER(c4_1,(/2/))
  TEST_WRAPPER(c4_2,(/2,1/))
  TEST_WRAPPER(c4_3,(/2,1,1/))
  TEST_WRAPPER(c4_4,(/2,1,1,1/))
  TEST_WRAPPER(c4_5,(/2,1,1,1,1/))
  TEST_WRAPPER(c4_6,(/2,1,1,1,1,1/))
  TEST_WRAPPER(c4_7,(/2,1,1,1,1,1,1/))
  TEST_WRAPPER(c8_1,(/2/))
  TEST_WRAPPER(c8_2,(/2,1/))
  TEST_WRAPPER(c8_3,(/2,1,1/))
  TEST_WRAPPER(c8_4,(/2,1,1,1/))
  TEST_WRAPPER(c8_5,(/2,1,1,1,1/))
  TEST_WRAPPER(c8_6,(/2,1,1,1,1,1/))
  TEST_WRAPPER(c8_7,(/2,1,1,1,1,1,1/))
  TEST_WRAPPER(lg_1,(/2/))
  TEST_WRAPPER(lg_2,(/2,1/))
  TEST_WRAPPER(lg_3,(/2,1,1/))
  TEST_WRAPPER(lg_4,(/2,1,1,1/))
  TEST_WRAPPER(lg_5,(/2,1,1,1,1/))
  TEST_WRAPPER(lg_6,(/2,1,1,1,1,1/))
  TEST_WRAPPER(lg_7,(/2,1,1,1,1,1,1/))

  call gator_finalize()
#ifdef HAVE_MPI
  call MPI_Finalize(ierr)
#endif
end program Fortran_Gator_Wrappers

#undef TEST_WRAPPER
