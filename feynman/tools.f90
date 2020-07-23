! Max Tegmark 171119, 190128-31, 190506, May 2020

	! Binary:
	! +: add
	! *: multiply
	! -: subtract
	! /: divide	(Put "D" instead of "/" in file, since f77 can't load backslash
	! Unary:
	!  >: increment (x -> x+1) 
	!  <: decrement (x -> x-1)
	!  ~: negate  	(x-> -x)
	!  \: invert    (x->1/x) (Put "I" instead of "\" in file, since f77 can't load backslash
	!  L: logaritm  (x-> ln(x)
	!  E: exponentiate (x->exp(x))
	!  S: sin:      (x->sin(x))       
	!  C: cos:      (x->cos(x))       
	!  A: abs:      (x->abs(x))
	!  N: arcsin    (x->arcsin(x))
	!  T: arctan    (x->arctan(x))
	!  R: sqrt	(x->sqrt(x))
        !  O: double    (x->2*x); note that this is the letter "O", not zero
	!  J: double+1  (x->2*x+1)
	! nonary: 
	!  0
	!  1
	!  P: pi
	real*8 function f(n,arities,ops,x) ! n=number of ops, x=arg vector
	implicit none
	integer nmax, n, i, j, arities(n), arity, lnblnk
	character*256 ops
	parameter(nmax=100)
	real*8 x(nmax), y, stack(nmax)
	character op
	!write(*,*) 'Evaluating function with ops = ',ops(1:n)
	!write(*,'(3f10.5,99i3)') (x(i),i=1,3), (arities(i),i=1,n)
	j = 0 ! Number of numbers on the stack
	do i=1,n
	  arity = arities(i)
	  op = ops(i:i)
	  if (arity.eq.0) then ! This is a nonary function
	    if (op.eq."0") then
              y = 0.
	    else if (op.eq."1") then
	      y = 1.
	    else if (op.eq."P") then
 	      y = 4.*atan(1.) ! pi
           else 
    	      y = x(ichar(op)-96)
	    end if
	  else if (arity.eq.1) then ! This is a unary function
	    if (op.eq.">") then
             y = stack(j) + 1
	    else if (op.eq."<") then
             y = stack(j) - 1
	    else if (op.eq."~") then
              y = -stack(j)
	    else if (op.eq."\") then
             y = 1./stack(j)
	    else if (op.eq."L") then
             y = log(stack(j))
	    else if (op.eq."E") then
             y = exp(stack(j))
	    else if (op.eq."S") then
             y = sin(stack(j))
	    else if (op.eq."C") then
             y =cos(stack(j))
	    else if (op.eq."A") then
             y = abs(stack(j))
	    else if (op.eq."N") then
             y = asin(stack(j))
	    else if (op.eq."T") then
             y = atan(stack(j))
	    else if (op.eq."O") then
             y = 2.*stack(j)
	    else if (op.eq."J") then
             y = 1+2.*stack(j)
	    else
             y = sqrt(stack(j))
	    end if
	  else ! This is a binary function
	  if (op.eq."+") then
	      y = stack(j-1)+stack(j)
	    else if (op.eq."-") then
	      y = stack(j-1)-stack(j)
	    else if (op.eq."*") then
	      y = stack(j-1)*stack(j)
	    else
	      y = stack(j-1)/stack(j)
	    end if
          end if
	  j = j + 1 - arity
          stack(j) = y  
	  ! write(*,'(9f10.5)') (stack(k),k=1,j)
	end do
	if (j.ne.1) stop 'DEATH ERROR: STACK UNBALANCED'
	f = stack(1)
        !write(*,'(9f10.5)') 666.,x(1),x(2),x(3),f
	return
	end

        subroutine multiloop(n,bases,i,done)
        ! Handles <n> nested loops with loop variables i(1),...i(n).
        ! Example: With n=3, bases=2, repeated calls starting with i=(000) will return
        ! 001, 010, 011, 100, 101, 110, 111, 000 (and done=.true. the last time).
        ! All it's doing is counting in mixed radix specified by the array <bases>.
        implicit none
        integer n, bases(n), i(n), k
        logical done
        done = .false.
        k = 1
555     i(k) = i(k) + 1
        if (i(k).lt.bases(k)) return
        i(k) = 0
        k = k + 1
        if (k.le.n) goto 555
        done = .true.
        return
        end

	real*8 function limit(x)
	implicit none
	real*8 x, xmax
	parameter(xmax=666.)
	if (abs(x).lt.xmax) then
 	  limit = x
	else
	  limit = sign(xmax,x)
	end if
	return
	end

	subroutine LoadMatrixTranspose(nd,n,mmax,m,A,f)
	! Reads the n x m matrix A from the file named f, stored as its transpose
	implicit none
	integer nd,mmax,n,m,j
	real*8 A(nd,mmax)
	character*256 f
	open(2,file=f,status='old')
	m = 0
555	m = m + 1
	if (m.gt.mmax) stop 'DEATH ERROR: m>mmax in LoadVectorTranspose'
	read(2,*,end=666) (A(j,m),j=1,n)
	goto 555
666	close(2)
	m = m - 1
	print *,m,' rows read from file ',f
	return
	end

	real*8 function mymedian(n,a)
        implicit none
        integer n,nmax, i
        parameter(nmax=10000000)
        real*8 a(n), b(nmax)
        if (n.gt.nmax) stop 'DEATH ERROR: n>nmax in mymedian'
        do i=1,n
          b(i) = a(i)
        end do
        call sort(n,b)
        i = nint((n+.5)/2)
        if (i.eq.0) i=1
        mymedian = b(i)
        return
        end

	subroutine permutation(n,iarr) ! Return a random permutation of the first n integer:
	integer iarr(n), idum, nmax, i
        parameter(nmax=10000000)
        real*8 arr(nmax), brr(nmax), ran1
        if (n.gt.nmax) stop "PERMUTATION DEATH ERROR: nmax TOO SMALL"
        idum = -666
        do i=1,n
          arr(i) = ran1(idum)
          brr(i) = i
        end do
        call sort2(n,arr,brr)
        do i=1,n
          iarr(i) = nint(brr(i))
        end do
	return
        end        
        
      SUBROUTINE sort(n,arr) ! Numerical Recipes Quicksort:
      INTEGER n,M,NSTACK
      REAL*8 arr(n)
      PARAMETER (M=7,NSTACK=50)
      INTEGER i,ir,j,jstack,k,l,istack(NSTACK)
      REAL*8 a,temp
      jstack=0
      l=1
      ir=n
1     if(ir-l.lt.M)then
        do 12 j=l+1,ir
          a=arr(j)
          do 11 i=j-1,1,-1
            if(arr(i).le.a)goto 2
            arr(i+1)=arr(i)
11        continue
          i=0
2         arr(i+1)=a
12      continue
        if(jstack.eq.0)return
        ir=istack(jstack)
        l=istack(jstack-1)
        jstack=jstack-2
      else
        k=(l+ir)/2
        temp=arr(k)
        arr(k)=arr(l+1)
        arr(l+1)=temp
        if(arr(l+1).gt.arr(ir))then
          temp=arr(l+1)
          arr(l+1)=arr(ir)
          arr(ir)=temp
        endif
        if(arr(l).gt.arr(ir))then
          temp=arr(l)
          arr(l)=arr(ir)
          arr(ir)=temp
        endif
        if(arr(l+1).gt.arr(l))then
          temp=arr(l+1)
          arr(l+1)=arr(l)
          arr(l)=temp
        endif
        i=l+1
        j=ir
        a=arr(l)
3       continue
          i=i+1
        if(arr(i).lt.a)goto 3
4       continue
          j=j-1
        if(arr(j).gt.a)goto 4
        if(j.lt.i)goto 5
        temp=arr(i)
        arr(i)=arr(j)
        arr(j)=temp
        goto 3
5       arr(l)=arr(j)
        arr(j)=a
        jstack=jstack+2
        if(jstack.gt.NSTACK) stop 'NSTACK too small in sort'
        if(ir-i+1.ge.j-l)then
          istack(jstack)=ir
          istack(jstack-1)=i
          ir=j-1
        else
          istack(jstack)=j-1
          istack(jstack-1)=l
          l=i
        endif
      endif
      goto 1
      END

      SUBROUTINE sort2(n,arr,brr)  ! Numerical Recipes Quicksort:
      INTEGER n,M,NSTACK
      REAL*8 arr(n),brr(n)
      PARAMETER (M=7,NSTACK=50)
      INTEGER i,ir,j,jstack,k,l,istack(NSTACK)
      REAL*8 a,b,temp
      jstack=0
      l=1
      ir=n
1     if(ir-l.lt.M)then
        do 12 j=l+1,ir
          a=arr(j)
          b=brr(j)
          do 11 i=j-1,1,-1
            if(arr(i).le.a)goto 2
            arr(i+1)=arr(i)
            brr(i+1)=brr(i)
11        continue
          i=0
2         arr(i+1)=a
          brr(i+1)=b
12      continue
        if(jstack.eq.0)return
        ir=istack(jstack)
        l=istack(jstack-1)
        jstack=jstack-2
      else
        k=(l+ir)/2
        temp=arr(k)
        arr(k)=arr(l+1)
        arr(l+1)=temp
        temp=brr(k)
        brr(k)=brr(l+1)
        brr(l+1)=temp
        if(arr(l+1).gt.arr(ir))then
          temp=arr(l+1)
          arr(l+1)=arr(ir)
          arr(ir)=temp
          temp=brr(l+1)
          brr(l+1)=brr(ir)
          brr(ir)=temp
        endif
        if(arr(l).gt.arr(ir))then
          temp=arr(l)
          arr(l)=arr(ir)
          arr(ir)=temp
          temp=brr(l)
          brr(l)=brr(ir)
          brr(ir)=temp
        endif
        if(arr(l+1).gt.arr(l))then
          temp=arr(l+1)
          arr(l+1)=arr(l)
          arr(l)=temp
          temp=brr(l+1)
          brr(l+1)=brr(l)
          brr(l)=temp
        endif
        i=l+1
        j=ir
        a=arr(l)
        b=brr(l)
3       continue
          i=i+1
        if(arr(i).lt.a)goto 3
4       continue
          j=j-1
        if(arr(j).gt.a)goto 4
        if(j.lt.i)goto 5
        temp=arr(i)
        arr(i)=arr(j)
        arr(j)=temp
        temp=brr(i)
        brr(i)=brr(j)
        brr(j)=temp
        goto 3
5       arr(l)=arr(j)
        arr(j)=a
        brr(l)=brr(j)
        brr(j)=b
        jstack=jstack+2
        if(jstack.gt.NSTACK)stop 'NSTACK too small in sort2'
        if(ir-i+1.ge.j-l)then
          istack(jstack)=ir
          istack(jstack-1)=i
          ir=j-1
        else
          istack(jstack)=j-1
          istack(jstack-1)=l
          l=i
        endif
      endif
      goto 1
      END

      ! Numerical Recipes random number generator:
      FUNCTION ran1(idum)
      INTEGER idum,IA,IM,IQ,IR,NDIV
      REAL*8 ran1,AM,EPS,RNMX
      PARAMETER (IA=16807,IM=2147483647,AM=1./IM,IQ=127773,IR=2836,NDIV=1+(IM-1)/32,EPS=1.2e-7,RNMX=1.-EPS)
      INTEGER j,k,iv(32),iy
      SAVE iv,iy
      DATA iv /32*0/, iy /0/
      if (idum.le.0.or.iy.eq.0) then
        idum=max(-idum,1)
        do 11 j=32+8,1,-1
          k=idum/IQ
          idum=IA*(idum-k*IQ)-IR*k
          if (idum.lt.0) idum=idum+IM
          if (j.le.32) iv(j)=idum
11      continue
        iy=iv(1)
      endif
      k=idum/IQ
      idum=IA*(idum-k*IQ)-IR*k
      if (idum.lt.0) idum=idum+IM
      j=1+iy/NDIV
      iy=iv(j)
      iv(j)=idum
      ran1=min(AM*iy,RNMX)
      return
      END

