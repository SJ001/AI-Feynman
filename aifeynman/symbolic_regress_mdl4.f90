	! Max Tegmark 171119, 190128-31, 190506, 200427-29
	! Loads templates.csv functions.dat and mystery.dat, returns winners.
        ! The mystery file contains 2n columns, where n is the number of variables: 
        !   first the input variables and then the components of grad f
        !   Success means that the candicate function's gradient is proportional to grad f.
        ! Rejects Pareto-dominated formulas not based on hard sup-norm cut, but using a 
        !   hypothesis-testing framework with a z-score z_n = sqrt(n)*(<b_n>-<b_best>)/sigma_best
	! scp -P2222 symbolic_regress.f euler@tor.mit.edu:FEYNMAN
	! COMPILATION: a f 'f77 -O3 -o symbolic_regress4_mdl.x symbolic_regress_mdl4.f |& more'
	! COMPILATION: a g 'f77 -O3 -o symbolic_regress4_mdl.x symbolic_regress_mdl4.f'
	! SAMPLE USAGE: call symbolic_regress4_mdl.x 7ops.txt arity2templates.txt gradmystery1.dat results.dat 10 0
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
	!
	! nonary: 
	!  0
	!  1
	!  P = pi
	!  a, b, c, ...: input variables for function (need not be listed in functions.dat)

	program symbolic_regress
	call go
	end
	
	subroutine go
	implicit none
	character*256 opsfile, templatefile, mysteryfile, outfile, usedfuncs
	character*60 comline, functions, ops, formula
	integer arities(21), nvar, nvarmax, nmax, lnblnk
	parameter(nvarmax=20, nmax=1000000)
	real*8 f, minloss, maxloss, rmsloss
        real*8 xy0(2*nvarmax,nmax), xy(2*nvarmax,nmax), gradf(nvarmax,nmax), gradfhat(nvarmax)
        real*8 epsilon, DL, nu, z
        real*8 lossbits, bitloss, meanbits, bestbits, bitmargin, bitsum, bitexcess, sigma, ev
	parameter(epsilon=1/2.**30)
	data arities /2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0/
	data functions /"+*-/><~\OJLESCANTR01P"/
      	integer nn(0:2), ii(nmax), kk(nmax), radix(nmax), iarr(nmax)
	integer ndata, i, j, n
	integer*8 nformulas, nevals
	logical done
	character*60 func(0:2), template

	nu = 5.
        bitmargin = 0. ! "Thickness" of pareto frontier; default 0
 	open(2,file='args.dat',status='old',err=666)
        read(2,*) opsfile, templatefile, mysteryfile, outfile, nu, bitmargin
 	write(*,'(1a24,f10.3)') 'Rejection threshold.....',nu
	write(*,'(1a24,f10.3)') 'Bit margin..............',bitmargin

        comline = 'head -1 '//mysteryfile(1:lnblnk(mysteryfile))//' | wc > qaz.dat'
        if (system(comline).ne.0) stop 'DEATH ERROR counting columns'
        open(2,file='qaz.dat')
        read(2,*) i, j
        close(2)
	nvar = j/2
	if (2*nvar.ne.j) stop 'DEATH ERROR: ODD NUMBER OF DATA COLUMNS'
	if (2*nvar.gt.nvarmax) stop 'DEATH ERROR: TOO MANY VARIABLES'
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
	call LoadMatrixTranspose(2*nvarmax,2*nvar,nmax,ndata,xy0,mysteryfile)	
	write(*,'(1a24,i8)') 'Number of examples......',ndata

	write(*,'(1a24)') 'Shuffling and normalizing mystery data....'
        call permutation(ndata,iarr)
	do i=1,ndata
          do j=1,nvar
            xy(j,i) = xy0(j,iarr(i))
            gradf(j,i) = xy0(nvar+j,iarr(i))
          end do
          call normalize(nvar,gradf(1,i))
        end do   	

	print *,'Searching for best fit...'
	nformulas = 0
        nevals = 0
	bestbits = 1.e6
        sigma = 1.d40 ! So that 1st function gets accepted
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
	  !print *, i,ii(i),kk(i),radix(i)
	end do
	done = .false.
	do while ((bestbits.gt.0).and.(.not.done))
	  nformulas = nformulas + 1
	  ! Analyze structure ii:
	  do i=1,n
	    ops(i:i) = func(ii(i))(1+kk(i):1+kk(i))
            !print *,'TEST ',i,ii(i), func(ii(i))
	  end do
	  !write(*,'(1f20.12,99i3)') bestbits, (ii(i),i=1,n), (kk(i),i=1,n)
          !write(*,'(1a24)') ops(1:n)
	  j = 1
          bitsum = 0.
          z = 0.
	  do while ((z.lt.nu).and.(j.le.ndata)) ! Keep going as long as you can't reject this formula
 	    nevals = nevals + 1
            call compute_gradfhat(n,ii,ops,nvar,xy(1,j),gradfhat)
            lossbits = bitloss(nvar,gradf(1,j),gradfhat)
            !write(*,'(1a,x,99f20.12)') ops(1:lnblnk(ops)),(xy(i,j),i=1,nvar), (gradf(i,j),i=1,nvar), (gradfhat(i),i=1,nvar), lossbits
            bitsum = bitsum + lossbits
            meanbits = bitsum/j
            bitexcess = meanbits - bestbits - bitmargin
            z = sqrt(1.*j)*bitexcess/sigma
    	    j = j + 1
	  end do
          !write(*,'(1f20.12,x,1a22,1i16,4f19.4)') meanbits, ops(1:n), nformulas, bestbits, z2, bitexcess
	  !if (meanbits.lt.bestbits+bitmargin) then ! We have a new point on the Pareto frontier
	  if (bitexcess.lt.0.) then ! We have a new point on the Pareto frontier
	    bestbits = min(meanbits,bestbits)
	    rmsloss = 0.
            maxloss = 0.
            sigma = 0.
	    do j=1,ndata
	      rmsloss = rmsloss + bitexcess**2
              if (maxloss.lt.bitexcess) maxloss = bitexcess
              sigma = (lossbits-meanbits)**2 
	    end do
	    rmsloss = sqrt(rmsloss/ndata)
            sigma = sqrt(sigma/ndata)
            !!! write(*,'(1f9.4)') sigma
	    DL  = log(1.*nformulas)/log(2.)
            ev = (1.*nevals)/nformulas
	    write(*,'(1f20.12,x,1a22,1i16,6f19.4)') meanbits, ops(1:n), nformulas, DL, DL+ndata*meanbits, rmsloss, maxloss, sigma, ev
	    write(3,'(1f20.12,x,1a22,1i16,6f19.4)') meanbits, ops(1:n), nformulas, DL, DL+ndata*meanbits, rmsloss, maxloss, sigma, ev
	    flush(3)
	  end if
!          if (ops(1:n)=="abb+S+R") then      
!            print *,"Eureka!"
!            write(*,'(1f20.12,x,1a22,1i16,6f19.4)') meanbits, ops(1:n), nformulas, DL, DL+ndata*meanbits, rmsloss, maxloss, sigma, ev
!            stop
!          end if
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

	! Returns log2(1-|vec1.vec2|)
        !|vec1.vec2|=1 when vectors are either parallel or antiparallel.
	real*8 function bitloss(n,vec1,vec2)
        integer n, i
        real*8 vec1(n), vec2(n), dot, loss
        dot = 0
        do i=1,n
          dot = dot + vec1(i)*vec2(i)
        end do
        loss = (1-abs(dot))*(2.**30)
 	if (.not.((loss.ge.0).or.(loss.le.0))) loss = 1.e30 ! This was a NaN :-)
        if (loss.gt.1) then 
          bitloss = 1.44269504089*log(loss)  ! = log2(newloss)
        else
          bitloss = 0.
        end if
        !print *,"KANIN",vec1,vec2,dot,loss,bitloss
        return
        end
        
        subroutine normalize(n,vec)
        implicit none
        integer n, i
        real*8 vec(n), norm
        norm = 0
        do i=1,n
          norm = norm + vec(i)**2
        end do 
	norm = max(1.d-30,norm)
        norm = 1/sqrt(norm)
        do i=1,n
          vec(i) = norm*vec(i)
        end do 
        !print *,'Normalized vector',vec
        return
        end

	! Returns unit vector pointing in direction of gradf
        subroutine compute_gradfhat(n,arities,ops,nvar,x,gradfhat) ! n=number of ops, x=arg vector
	implicit none
	integer nmax, n, arities(n), nvar, i, j
	character*60 ops
        real*8 f, epsilon
	parameter(nmax=100,epsilon=1.d-10)
	real*8 x(nvar), x1(nmax), x2(nmax), gradfhat(nvar)
	do i=1,nvar
          x1(i) = x(i)
          x2(i) = x(i)
        end do
        do i=1,nvar
          x1(i) = x(i)-epsilon
          x2(i) = x(i)+epsilon
          gradfhat(i) = (f(n,arities,ops,x2)-f(n,arities,ops,x1))/(2*epsilon)
          !print *,'###',ops,(x1(j),j=1,nvar),gradfhat(i), f(n,arities,ops,x2),f(n,arities,ops,x1),norm
          !read *
          x1(i) = x(i)
          x2(i) = x(i)
        end do
        call normalize(nvar,gradfhat)
	return
        end
        
        include "tools.f90"
