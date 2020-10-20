	! Max Tegmark 171119, 190128-31, 190506, 200427-29
	! Loads templates.csv functions.dat and mystery.dat, returns winners.
        ! Rejects Pareto-dominated formulas not based on hard sup-norm cut, but using a 
        !   hypothesis-testing framework with a z-score z_n = sqrt(n)*(<b_n>-<b_best>)/sigma_best
	! scp -P2222 symbolic_regress.f euler@tor.mit.edu:FEYNMAN
	! COMPILATION: a f 'f77 -O3 -o symbolic_regress_mdl2.x symbolic_regress_mdl2.f |& more'
	! SAMPLE USAGE: call symbolic_regress_mdl2.x 7ops.txt arity2templates.txt mystery2.dat results.dat 10 0
        !               call symbolic_regress_mdl2.x 6ops.txt arity2templates.txt mysteryB3.dat results.dat 10 0 (takes a few minutes)
	! 		call symbolic_regress_mdl2.x 14ops.txt arity2templates.txt mystery.dat results.dat 10 0
	! 		call symbolic_regress_mdl2.x 14ops.txt arity2templates.txt mystery.dat results.dat 1000 0 (if skips over correct formula)
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
	parameter(nvarmax=20, nmax=5000000)
	real*8 f, newloss, minloss, maxloss, rmsloss, limit
        real*8 xy0(nvarmax+1,nmax), xy(nvarmax+1,nmax), offset(nmax), offst, bestoffset
        real*8 epsilon, DL, nu, z
        real*8 lossbits, bitmean, bitsdev, bestbits, bitmargin, sigma, bitexcess, ev
	parameter(epsilon=1/2.**30)
	data arities /2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0/
	data functions /"+*-/><~\OJLESCANTR01P"/
      	integer nn(0:2), ii(nmax), kk(nmax), radix(nmax), iarr(nmax)
	integer ndata, i, j, jtest, n
	integer*8 nformulas, nevals
	logical done, rejected
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
	call LoadMatrixTranspose(nvarmax+1,nvar+1,nmax,ndata,xy0,mysteryfile)	
	write(*,'(1a24,i8)') 'Number of examples......',ndata

	write(*,'(1a24)') 'Shuffling mystery data....'
        call permutation(ndata,iarr)
	do i=1,ndata
          do j=1,nvar+1
            xy(j,i) = xy0(j,iarr(i))
          end do
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
	end do
	done = .false.
	do while ((bestbits.gt.0).and.(.not.done))
	  nformulas = nformulas + 1
	  ! Analyze structure ii:
	  do i=1,n
	    ops(i:i) = func(ii(i))(1+kk(i):1+kk(i))
	  end do
	  j = 1
 	  jtest = 2  ! Will test after j=2, 3, 5, 9, 17, ... data points
	  rejected = .false.
          do while ((.not.rejected).and.(j.le.ndata)) ! Keep going as long as you can't reject this formula
 	    nevals = nevals + 1
            offst = xy(nvar+1,j) - f(n,ii,ops,xy(1,j))
            rejected = (.not.((offst.ge.0).or.(offst.le.0))) ! This was a NaN, so reject the formula :-)
            !if (rejected) print *,"NaN!"
            if (rejected) exit
            rejected = abs(offst).gt.(1./epsilon) ! Otherwise numerical cancellation can masquerade as successss
            !if (rejected) print *,"Infinity!"
            if (rejected) exit     
	    offset(j) = offst    
            if (j.ge.jtest) then ! Time for another test                
                call analyze_offset(j,offset,epsilon,bestoffset,bitmean,bitsdev)
                bitexcess = bitmean - bestbits - bitmargin
                z = sqrt(1.*j)*bitexcess/sigma ! This sigma is for previous winner, not for this candidate
                rejected = (z.gt.nu)
                jtest = min(2*jtest-1,ndata)
            end if  
    	    j = j + 1
	  end do
	  if (.not.rejected.and.(bitexcess.lt.0.)) then ! We have a new point on the Pareto frontier
	    bestbits = min(bitmean,bestbits)
	    rmsloss = 0.
            maxloss = 0.
            sigma = 0.
	    do j=1,ndata
              newloss = abs(xy(nvar+1,j) - f(n,ii,ops,xy(1,j)) - bestoffset)
	      rmsloss = rmsloss + newloss**2
              if (maxloss.lt.newloss) maxloss = newloss
	    end do
	    rmsloss = sqrt(rmsloss/ndata)
            sigma = bitsdev
	    DL  = log(1.*nformulas)/log(2.)
            ev = (1.*nevals)/nformulas
	    write(*,'(2f20.12,x,1a22,1i16,6f19.4)') bitmean, limit(bestoffset), ops(1:n), nformulas, DL, &
                DL+ndata*bitmean, rmsloss, maxloss, bitsdev, ev
	    write(3,'(2f20.12,x,1a22,1i16,6f19.4)') bitmean, limit(bestoffset), ops(1:n), nformulas, DL, &
                DL+ndata*bitmean, rmsloss, maxloss, bitsdev, ev
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

	subroutine analyze_offset(n,offset,epsilon,median,bitmean,bitsdev) ! Check how much an array departs from its median
	implicit none
        integer n, i
        real*8 offset(n), epsilon, bitmean, bitsdev
        real*8 median, mymedian, x, bits, sum1, sum2
        median = mymedian(n,offset)
        sum1 = 0.
        sum2 = 0.
        do i=1,n
          x = abs(offset(i)-median)/epsilon        
          if (x.gt.1) then 
            bits = 1.44269504089*log(x)  ! = log2(x)
          else
            bits = 0.
          end if
	  sum1 = sum1 + bits
          sum2 = sum2 + bits*bits          
        end do
        bitmean = sum1/n
        bitsdev = sqrt(abs(sum2/n-bitmean**2))
     	return
        end

	include "tools.f90"
