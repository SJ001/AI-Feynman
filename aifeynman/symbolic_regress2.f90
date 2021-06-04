	! Max Tegmark 171119, 190128-31, 190218
	! Same as symbolic_regress2.f except that it fits for the symbolic formula times an arbitrary constant.
	! Loads templates.csv functions.dat and mystery.dat, returns winner.
	! scp -P2222 symbolic_regress2.f euler@tor.mit.edu:FEYNMAN
	! COMPILATION: a f 'f77 -O3 -o symbolic_regress2.x symbolic_regress2.f |& more'
	! SAMPLE USAGE: call symbolic_regress2.x 4ops.txt arity2templates.txt mysteryB3.dat results.dat
	! functions.dat contains a single line (say "0>+*-/") with the single-character symbols 
        ! that will be used, drawn from this list: 
	!
	! Binary:
	! +: add
	! *: multiply
	! -: subtract
	! /: divide	(Put "D" instead of "/" in file, since f77 can't load backslash
	!
	! Unary:
	!  >: increment (x -> x+1) 
	!  <: decrement (x -> x-1)
	!  ~: negate  	(x-> -x)
	!  \: invert    (x->1/x) (Put "I" instead of "\" in file, since f77 can't load backslash
	!  L: logaritm: (x-> ln(x)
	!  E: exponentiate (x->exp(x))
	!  S: sin:      (x->sin(x))       
	!  C: cos:      (x->cos(x))       
	!  A: abs:      (x->abs(x))
	!  N: arcsin:   (x->arcsin(x))
	!  T: arctan:   (x->arctan(x))
	!  R: sqrt	(x->sqrt(x))
	!
	! nonary: 
	!  0
	!  1
	!  a, b, c, ...: input variables for function (need not be listed in functions.dat)

	program symbolic_regress
	call go
	end
	
	subroutine go
	implicit none
	character*256 opsfile, templatefile, mysteryfile, outfile, usedfuncs
	character*60 comline, functions, ops, formula
	integer arities(21), nvar, nvarmax, nmax, lnblnk
	parameter(nvarmax=20, nmax=10000000)
	real*8 f, newloss, minloss, maxloss, rmsloss, xy(nvarmax+1,nmax), epsilon
	real*8 ymax, prefactor, DL, DL2, DL3, limit
	parameter(epsilon=0.00001)
	data arities /2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0/
	data functions /"+*-/><~\OJLESCANTR01P"/
      	integer nn(0:2), ii(nmax), kk(nmax), radix(nmax)
	integer ndata, i, j, n, jmax
	integer*8 nformulas
	logical done
	character*60 func(0:2), template

	open(2,file='args.dat',status='old',err=666)
	read(2,*) opsfile, templatefile, mysteryfile, outfile
	close(2)

        comline = 'head -1 '//mysteryfile(1:lnblnk(mysteryfile))//' | wc > qaz.dat'
        if (system(comline).ne.0) stop 'DEATH ERROR counting columns'
        open(2,file='qaz.dat')
        read(2,*) i, nvar
        close(2)
	nvar = nvar - 1
	if (nvar.gt.nvarmax) stop 'DEATH ERROR: TOO MANY VARIABLES'
	write(*,'(1a24,i8)') 'Number of variables.....',nvar

	open(2,file=opsfile,status='old',err=668)
	read(2,*) usedfuncs
	close(2)
	nn(0)=0
	nn(1)=0
	nn(2)=0
	do i=1,lnblnk(usedfuncs) 
	  if (usedfuncs(i:i).eq.'D') usedfuncs(i:i)='/'
	  if (usedfuncs(i:i).eq.'I') usedfuncs(i:i)='\'
	  j = index(functions,usedfuncs(i:i))
	  if (j.eq.0) then
            print *,'DEATH ERROR: Unknown function requested: ',usedfuncs(i:i)
	    stop
	  else
	    nn(arities(j)) = nn(arities(j)) + 1
	    func(arities(j))(nn(arities(j)):nn(arities(j))) = functions(j:j)
         end if
	end do
	! Add nonary ops to retrieve each of the input variables:
	do i=1,nvar
          nn(0) = nn(0) + 1
	  func(0)(nn(0):nn(0)) = char(96+i)
	end do
	write(*,'(1a24,1a22)') 'Functions used..........',usedfuncs(1:lnblnk(usedfuncs))
	do i=0,2
	  write(*,*) 'Arity ',i,': ',func(i)(1:nn(i))
	end do

	write(*,'(1a24)') 'Loading mystery data....'
	call LoadMatrixTranspose(nvarmax+1,nvar+1,nmax,ndata,xy,mysteryfile)	
	write(*,'(1a24,i8)') 'Number of examples......',ndata
	! Find max(abs(y)) to use for normalization estimation (crucial to avoid data point where y~0):
	jmax=1
	ymax = abs(xy(1,nvar+1))
	do j=2,ndata
	  if (ymax < abs(xy(nvar+1,j))) then 
             ymax = abs(xy(nvar+1,j))
	     jmax = j
	  end if
	end do
	print *,'Mystery data has largest magnitude ',ymax,' at j=',jmax
	print *,'Searching for best fit...'	
	nformulas = 0
	minloss = 1.e6
	template = ''
	ops='===================='
	open(2,file=templatefile,status='old',err=670)
	open(3,file=outfile)
555	read(2,'(1a60)',end=665) template
	n = lnblnk(template)
	!print *,"template:",template(1:n),"#####"
	do i=1,n
	  ii(i) = ichar(template(i:i))-48
	  radix(i) = nn(ii(i))
	  kk(i) = 0
	end do
	done = .false.
	do while ((minloss.gt.epsilon).and.(.not.done))
	  nformulas = nformulas + 1
	  ! Analyze structure ii:
	  do i=1,n
	    ops(i:i) = func(ii(i))(1+kk(i):1+kk(i))
            !print *,'TEST ',i,ii(i), func(ii(i))
	  end do
	  !write(*,'(1f20.12,99i3)') minloss, (ii(i),i=1,n), (kk(i),i=1,n)
          !write(*,'(1a24)') ops(1:n)

	  prefactor = xy(nvar+1,jmax)/f(n,ii,ops,xy(1,jmax))
	  j = 1
	  maxloss = 0.
	  do while ((maxloss.lt.minloss).and.(j.le.ndata))
	    newloss = abs(xy(nvar+1,j) - prefactor*f(n,ii,ops,xy(1,j)))
            !!!!!print *,'newloss: ',j,newloss,xy(nvar,j),f(n,ii,ops,xy(1,j))
	    if (.not.((newloss.ge.0).or.(newloss.le.0))) newloss = 1.e30 ! This was a NaN :-)
	    if (maxloss.lt.newloss) maxloss = newloss
	    j = j + 1
	  end do
	  if (maxloss.lt.minloss) then ! We have a new best fit
	    minloss = maxloss
	    rmsloss = 0.
	    do j=1,ndata
	      rmsloss = rmsloss + (xy(nvar+1,j) - prefactor*f(n,ii,ops,xy(1,j)))**2
	    end do
	    rmsloss = sqrt(rmsloss/ndata)
	    DL  = log(nformulas*max(1.,minloss/epsilon))/log(2.)
	    DL2 = log(nformulas*max(1.,minloss/1.e-15))/log(2.)
	    DL3 = (log(1.*nformulas) + sqrt(1.*ndata)*log(max(1.,rmsloss/1.e-15)))/log(2.)
	    write(*,'(2f20.12,x,1a22,1i16,4f19.4)') limit(minloss), limit(prefactor), ops(1:n), nformulas, rmsloss, DL, DL2, DL3
	    write(3,'(2f20.12,x,1a22,1i16,4f19.4)') limit(minloss), limit(prefactor), ops(1:n), nformulas, rmsloss, DL, DL2, DL3
	    flush(3)
	  end if
	  call multiloop(n,radix,kk,done) 
	end do
	goto 555
665	close(3)
	close(2)
	print *,'All done: results in ',outfile
	return
666	stop 'DEATH ERROR: missing file args.dat'
668	print *,'DEATH ERROR: missing file ',opsfile(1:lnblnk(opsfile))
	stop
670	print *,'DEATH ERROR: missing file ',templatefile(1:lnblnk(templatefile))
	stop
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

	real*8 function f(n,arities,ops,x) ! n=number of ops, x=arg vector
	implicit none
	integer nmax, n, i, j, arities(n), arity, lnblnk
	character*60 ops
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

	subroutine LoadMatrixTranspose(nd,n,mmax,m,A,f)
	! Reads the n x m matrix A from the file named f, stored as its transpose
	implicit none
	integer nd,mmax,n,m,j
	real*8 A(nd,mmax)
	character*60 f
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

