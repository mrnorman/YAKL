      program test_penta
      implicit none

! test the pentadiagonal and cyclic pentadiagonal solvers

! declare
      integer            :: i
      integer, parameter :: np = 7
      real*8             :: a(np),b(np),c(np),d(np),e(np),u(np),rhs(np), &
                            x(np),cp1,cp2,cp3,cp4,cp5,cp6, &
                            amat(np,np), xwant(np)

! formats
 01   format(1x,1p7e10.2)
 02   format(1x,i4,1pe10.2)
 03   format(1x,i4,1p3e10.2)


! set the matrix
      a = (/ 0.0d0, 0.0d0, 0.2d0, 0.9d0, 4.0d0, 2.2d0, 1.1d0/)
      b = (/ 0.0d0, 3.6d0, 2.8d0, 3.1d0, 6.7d0, 1.2d0, 0.1d0/)
      c = (/ 4.1d0, 2.2d0, 6.2d0, 8.5d0, 3.8d0, 3.7d0, 2.1d0/)
      d = (/ 0.4d0, 1.0d0, 5.0d0, 4.9d0, 2.3d0, 5.1d0, 0.0d0/)
      e = (/ 0.5d0, 6.1d0, 2.9d0, 4.5d0, 0.7d0, 0.0d0, 0.0d0/)

      xwant = (/ 1.0d0, 2.0d0,  3.0d0,  4.0d0, 5.0d0,  6.0d0, 7.0d0  /)


! put in regular matrix format for ease of manipulation
      amat = 0.0d0
      do i=3,np
       amat(i,i-2) = a(i)
      enddo
      do i=2,np
       amat(i,i-1) = b(i)
      enddo
      do i=1,np
       amat(i,i)   = c(i)
      enddo
      do i=1,np-1
       amat(i,i+1) = d(i)
      enddo
      do i=1,np-2
       amat(i,i+2) = e(i)
      end do


! write the matrix
      write(6,*)
      write(6,*) 'pentadiagonal matrix:'
      write(6,01) (amat(i,:), i=1,np)

! set the right hand side based on a known solution
      rhs = matmul(amat, xwant)

      write(6,*)
      write(6,*) 'setting rhs to:'
      write(6,01) rhs


! solve the cyclic pentadiagonal matrix
      call pentadiagonal(a,b,c,d,e,rhs,u,np)



! write the solution
      write(6,*)
      write(6,*) 'solution'
      write(6,02) (i, u(i), i=1,np)


! check the solution
      x = matmul(amat, u)

      write(6,*)
      write(6,*) 'residuals in x-rhs column should be small:'
      write(6,*) '   i   x         rhs       x-rhs'
      write(6,03) (i, x(i), rhs(i), x(i)-rhs(i), i=1,np)




! now test the cyclic pentadiagonal solver
!
!   x    x    x    0    0  ... 0  cp1  cp2
!   x    x    x    x    0  ... 0  0    cp3
!   .    .    .    .    .  .   .  .    .
!   cp4  0    0    ...  0  0   x  x    x
!   cp5  cp6  0    ...  0  x   x  x    x
!

      cp1 = 0.2d0 ; cp2 = 0.5d0 ; cp3 = 0.4d0 ; cp4 = 0.1d0 ; cp5 = 0.9d0; cp6 = 1.0d0
      amat(1,np-1) = cp1
      amat(1,np)   = cp2
      amat(2,np)   = cp3
      amat(np-1,1) = cp4
      amat(np,1)   = cp5
      amat(np,2)   = cp6


! write the matrix
      write(6,*)
      write(6,*)
      write(6,*) 'cyclic heptadiagonal matrix:'
      write(6,01) (amat(i,:), i=1,np)


! set the right hand side based on a known solution
      rhs = matmul(amat, xwant)

      write(6,*)
      write(6,*) 'setting rhs to:'
      write(6,01) rhs


! solve the cyclic pentadiagonal matrix
      call cyclic_pentadiagonal(a,b,c,d,e,rhs,cp1,cp2,cp3,cp4,cp5,cp6,u,np)


! write the solution
      write(6,*)
      write(6,*) 'solution'
      write(6,02) (i, u(i), i=1,np)


! check the solution
      x = matmul(amat, u)

      write(6,*)
      write(6,*) 'residuals in x-rhs column should be small:'
      write(6,*) '   i   x         rhs       x-rhs'
      write(6,03) (i, x(i), rhs(i), x(i)-rhs(i), i=1,np)


      end program test_penta




      subroutine pentadiagonal(a,b,c,d,e,f,u,n)
      implicit none

! solves for a vector u of length n in the pentadiagonal linear system
!  a_i u_(i-2) + b_i u_(i-1) + c_i u_i + d_i u_(i+1) + e_i u_(i+2) = f_i
! input are the a, b, c, d, e, and f and they are not modified

! in its clearest incarnation, this algorithm uses three storage arrays
! called p, q and r. here, the solution vector u is used for r, cutting
! the extra storage down to two arrays.

! declare the pass
      integer    :: n
      real*8     :: a(n),b(n),c(n),d(n),e(n),f(n),u(n)

! local variables
      integer, parameter :: nmax=500  ! maximum number of equations
      integer            :: i
      real*8             :: p(nmax),q(nmax),bet,den


! initialize elimination and backsubstitution arrays
      if (c(1) .eq. 0.0)  stop 'eliminate u2 trivially'
      bet  = 1.0d0/c(1)
      p(1) = -d(1) * bet
      q(1) = -e(1) * bet
      u(1) =  f(1) * bet

      bet = c(2) + b(2)*p(1)
      if (bet .eq. 0.0) stop 'bet singular in pentadiagonal'
      bet  = -1.0d0/bet
      p(2) = (d(2) + b(2)*q(1)) * bet
      q(2) = e(2) * bet
      u(2) = (b(2)*u(1) - f(2)) * bet


! reduce to upper triangular
      do i=3,n
       bet = b(i) + a(i) * p(i-2)
       den = c(i) + a(i)*q(i-2) + bet*p(i-1)
       if (den .eq. 0.0) stop 'den singular in pentadiagonal'
       den = -1.0d0/den
       p(i) = (d(i) + bet*q(i-1)) * den
       q(i) = e(i) * den
       u(i) = (a(i)*u(i-2) + bet*u(i-1) - f(i)) * den
      enddo

! backsubstitution
      u(n-1) = u(n-1) + p(n-1) * u(n)
      do i=n-2,1,-1
       u(i) = u(i) + p(i) * u(i+1) + q(i) * u(i+2)
      enddo
      return
      end subroutine pentadiagonal






      subroutine cyclic_pentadiagonal(a,b,c,d,e,f,cp1,cp2,cp3,cp4,cp5,cp6,x,n)
      implicit none
      save

! solves for x(1:n) a pentadiagonal matrix with nonzero entries in the
! lower left and upper right corners of the matrix:

!   x    x    x    0    0  ... 0  cp1  cp2
!   x    x    x    x    0  ... 0  0    cp3
!   .    .    .    .    .  .   .  .    .
!   cp4  0    0    ...  0  0   x  x    x
!   cp5  cp6  0    ...  0  x   x  x    x

! the woodbury formula is applied to the pentadiagonal matrix.


! declare the pass
      integer   ::  n
      real*8    ::  a(n),b(n),c(n),d(n),e(n),f(n), &
                    cp1,cp2,cp3,cp4,cp5,cp6,x(n)


! local variables
      integer, parameter :: nmax=500  ! maximum number of equations
      integer            :: i,j,k, indx(nmax)
      real*8             :: u(nmax,4),v(nmax,4),z(nmax,4), &
                            r(nmax),s(nmax),y(nmax), &
                            h(4,4),p(4,4),sum


! initialize
!      if (n .le. 5) stop 'n < 2 in routine cyclic_pentadiagonal'
!      if (n .gt. nmax) stop 'n > nmax in routine cyclic_pentadiagonal'

      u = 0.0d0
      v = 0.0d0
      z = 0.0d0

      u(1,1)   = 1.0d0
      u(2,2)   = 1.0d0
      u(n-1,3) = 1.0d0
      u(n,4)   = 1.0d0

      v(n-1,1) = cp1
      v(n,1)   = cp2
      v(n,2)   = cp3
      v(1,3)   = cp4
      v(1,4)   = cp5
      v(2,4)   = cp6


! solve the auxillary systems
! recipies equation 2.7.17 and 2.7.20
      call pentadiagonal(a,b,c,d,e,u(1,1),z(1,1),n)
      call pentadiagonal(a,b,c,d,e,u(1,2),z(1,2),n)
      call pentadiagonal(a,b,c,d,e,u(1,3),z(1,3),n)
      call pentadiagonal(a,b,c,d,e,u(1,4),z(1,4),n)
      call pentadiagonal(a,b,c,d,e,f,y,n)


! form the 4x4 matrix h
! recipies equation 2.7.19

      do j=1,4
       do i=1,4
        sum = 0.0d0
        do k=1,n
         sum = sum + v(k,j) * z(k,i)
        end do
        p(j,i) = sum
       end do
      enddo
      do i=1,4
       p(i,i) = p(i,i) + 1.0d0
      enddo
      call luinv(p,4,4,indx,h)

! form the solution
! recipe equation 2.7.21

      do j=1,4
       r(j) = 0.0d0
       do k=1,n
        r(j) = r(j) + v(k,j) * y(k)
       enddo
      enddo

      do j=1,4
       s(j) = 0.0d0
       do k=1,4
        s(j) = s(j) + h(j,k) * r(k)
       enddo
      enddo

      do j=1,n
       sum = 0.0d0
       do k=1,4
        sum = sum + z(j,k) * s(k)
       enddo
       x(j) = y(j) - sum
      enddo

      return
      end subroutine cyclic_pentadiagonal




      subroutine luinv(a,n,np,indx,y)
      implicit none

! this routine takes as input the n by n matrix a, of physical dimension
! np by np and on output fills y with the inverse of a
! 
! declare the pass
      integer :: n,np,indx(np)
      real*8  :: a(np,np),y(np,np)

! local variables
      integer :: i,j
      real*8  :: d


! set y to the identity matrix
      y = 0.0d0
      do i=1,n
       y(i,i) = 1.0d0
      enddo

! decomp and backsubstitute each column
      call ludcmp(a,n,np,indx,d)
      do j=1,n
       call lubksb(a,n,np,indx,y(1,j))
      enddo
      return
      end



      subroutine ludcmp(a,n,np,indx,d)
      implicit none

! given the matrix a(n,n), with physical dimsnsions a(np,ap) this routine
! replaces a by the lu decompostion of a row-wise permutation of itself.
! input are a,n,np. output is a, indx which records the row
! permutations effected by the partial pivoting, and d which is 1 if
! the number of interchanges is even, -1 if odd.
! use routine lubksb to solve a system of linear equations.
! 
! nmax is the largest expected value of n
! 
! declare the pass
      integer            :: n,np,indx(np)
      real*8             :: a(np,np),d


! local variables
      integer, parameter :: nmax = 100   ! maximum number of equations
      integer            :: i,j,k,imax
      real*8             :: vv(nmax),aamax,sum,dum
      real*8, parameter  :: tiny=1.0d-20

! bullet check
      if (n .gt. nmax) stop 'n > nmax in routine ludcmp'

! vv stores the implicit scaling of each row
! loop over the rows to get the scaling information
      d = 1.0d0
      do i=1,n
       aamax = 0.0d0
       do j=1,n
        if (abs(a(i,j)) .gt. aamax) aamax = abs(a(i,j))
       enddo
       if (aamax .eq. 0.0) stop 'singular matrix in ludcmp'
       vv(i) = 1.0d0/aamax
      enddo

! for each column apply crouts method; see equation 2.3.12
      do j=1,n
       do i=1,j-1
        sum = a(i,j)
        do k=1,i-1
         sum = sum - a(i,k)*a(k,j)
        enddo
        a(i,j) = sum
       enddo

! find the largest pivot element
       aamax = 0.0d0
       do i=j,n
        sum = a(i,j)
        do k=1,j-1
         sum = sum - a(i,k)*a(k,j)
        enddo
        a(i,j) = sum
        dum = vv(i)*abs(sum)
        if (dum .ge. aamax) then
         imax  = i
         aamax = dum
        end if
       enddo

! if we need to interchange rows
       if (j .ne. imax) then
        do k=1,n
         dum       = a(imax,k)
         a(imax,k) = a(j,k)
         a(j,k)    = dum
        enddo
        d          = -d
        vv(imax)   = vv(j)
       end if

! divide by the pivot element
       indx(j) = imax
       if (a(j,j) .eq. 0.0) a(j,j) = tiny
       if (j .ne. n) then
        dum = 1.0d0/a(j,j)
        do i=j+1,n
         a(i,j) = a(i,j)*dum
        enddo
       end if

! and go back for another column of crouts method
      enddo
      return
      end




      subroutine lubksb(a,n,np,indx,b)
      implicit none

! solves a set of n linear equations ax=b. a is input in its lu decomposition
! form, determined by the routine above ludcmp. indx is input as the
! permutation vector also returned by ludcmp. b is input as the right hand
! side vector and returns with the solution vector x.
! a,n ans np are not modified by this routine and thus can be left in place
! for successive calls (i.e matrix inversion)
! 
! declare the pass
      integer    :: n,np,indx(np)
      real*8     :: a(np,np),b(np)

! local variables
      integer    :: i,ii,j,ll
      real*8     :: sum

! when ii is > 0, ii becomes the index of the first nonzero element of b
! this is forward substitution of equation 2.3.6, and unscamble in place
      ii = 0
      do i=1,n
       ll = indx(i)
       sum = b(ll)
       b(ll) = b(i)
       if (ii .ne. 0) then
        do j=ii,i-1
         sum = sum - a(i,j) * b(j)
        enddo

! nonzero element was found, so dos the sums in the loop above
       else if (sum .ne. 0.0) then
        ii  = i
       end if
       b(i) = sum
      enddo

! back substitution equation 2.3.7
      do i = n,1,-1
       sum = b(i)
       if (i .lt. n) then
        do j=i+1,n
         sum = sum - a(i,j) * b(j)
        enddo
       end if
       b(i) = sum/a(i,i)
      enddo
      return
      end





