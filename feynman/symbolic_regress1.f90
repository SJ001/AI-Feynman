	! Max Tegmark 171119, 190128-31, 190506
	! Loads templates.csv functions.dat and mystery.dat, returns winner.
	! scp -P2222 symbolic_regress1.f euler@tor.mit.edu:FEYNMAN
	! COMPILATION: a f 'f77 -O3 -o symbolic_regress1.x symbolic_regress1.f |& more'
	! SAMPLE USAGE: call symbolic_regress1.x 10ops.txt arity2templates.txt mystery_constant.dat results.dat
	! functions.dat contains a single line (say "0>+*-/") with the single-character symbols 
        ! that will be used, drawn from this list: 
	!
	! Binary:
	! +: add
	! *: multiply
	! -: subtract
	! /: divide	(Put "D" instead of "/" in file, since f77 can't load backslash
	! Unary:
        !  O: double    (x->2*x); note that this is the letter "O", not zero
	!  J: double+1  (x->2*x+1)
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
	! nonary: 
	!  0
	!  1
	!  P: pi
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
	real*8 f, newloss, minloss, maxloss, rmsloss, xy(nvarmax+1,nmax), epsilon, DL, DL2, DL3
	parameter(epsilon=0.00000001)
	data arities /2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0/
	data functions /"+*-/><~\OJLESCANTR01P"/
      	integer nn(0:2), ii(nmax), kk(nmax), radix(nmax)
	integer ndata, i, j, n
	integer*8 nformulas
	logical done
	character*60 func(0:2), template

	open(2,file='args.dat',status='old',err=666)
	read(2,*) opsfile, templatefile, mysteryfile, outfile
	close(2)

        nvar = 0
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
	  !print *,'ASILOMAR ', i,ii(i),kk(i),radix(i)
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
	  j = 1
	  maxloss = 0.
	  do while ((maxloss.lt.minloss).and.(j.le.ndata))
	    newloss = abs(xy(nvar+1,j) - f(n,ii,ops,xy(1,j)))
            !!!!!print *,'newloss: ',j,newloss,xy(nvar,j),f(n,ii,ops,xy(1,j))
	    if (.not.((newloss.ge.0).or.(newloss.le.0))) newloss = 1.e30 ! This was a NaN :-)
	    if (maxloss.lt.newloss) maxloss = newloss
	    j = j + 1
	  end do
	  if (maxloss.lt.minloss) then ! We have a new best fit
	    minloss = maxloss
	    rmsloss = 0.
	    do j=1,ndata
	      rmsloss = rmsloss + (xy(nvar+1,j) - f(n,ii,ops,xy(1,j)))**2
	    end do
	    rmsloss = sqrt(rmsloss/ndata)
	    DL  = log(nformulas*max(1.,rmsloss/epsilon))/log(2.)
	    DL2 = log(nformulas*max(1.,rmsloss/1.e-15))/log(2.)
	    DL3 = (log(1.*nformulas) + sqrt(1.*ndata)*log(max(1.,rmsloss/1.e-15)))/log(2.)
	    write(*,'(1f20.12,x,1a22,1i16,4f19.4)') minloss, ops(1:n), nformulas, rmsloss, DL, DL2, DL3
	    write(3,'(1f20.12,x,1a22,1i16,4f19.4)') minloss, ops(1:n), nformulas, rmsloss, DL, DL2, DL3
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

        include 'tools.f90'
