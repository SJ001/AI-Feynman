gfortran -ffixed-line-length-none -O3 -o symbolic_regress1.x symbolic_regress1.f
gfortran -ffixed-line-length-none -O3 -o symbolic_regress2.x symbolic_regress2.f
gfortran -ffixed-line-length-none -O3 -o symbolic_regress3.x symbolic_regress3.f
gfortran -ffixed-line-length-none -O3 -o symbolic_regress_mdl2.x symbolic_regress_mdl2.f
gfortran -ffixed-line-length-none -O3 -o symbolic_regress_mdl3.x symbolic_regress_mdl3.f

chmod 555 brute_force_oneFile_v1.scr
chmod 555 brute_force_oneFile_v2.scr
chmod 555 brute_force_oneFile_v3.scr
chmod 555 brute_force_oneFile_mdl_v2.scr
chmod 555 brute_force_oneFile_mdl_v3.scr

