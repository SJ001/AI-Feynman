import numpy as np
import os
from S_run_bf_polyfit import run_bf_polyfit

def get_acos(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=4):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
            variables = np.column_stack((variables,v))
        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))

        dt = np.column_stack((variables,np.arccos(f_dependent)))
        np.savetxt(pathdir_write_to+filename,dt)     
                
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "acos")

    except:
        return PA

    return PA

def get_asin(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=4):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
            variables = np.column_stack((variables,v))
        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))

        dt = np.column_stack((variables,np.arcsin(f_dependent)))
        np.savetxt(pathdir_write_to+filename,dt)       
                
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "asin")

    except:
        return PA

    return PA

def get_atan(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=4):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
            variables = np.column_stack((variables,v))
        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))

        dt = np.column_stack((variables,np.arctan(f_dependent)))
        np.savetxt(pathdir_write_to+filename,dt) 
                
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "atan")                

    except:
        return PA

    return PA


def get_cos(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=4):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
            variables = np.column_stack((variables,v))
        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))

        dt = np.column_stack((variables,np.cos(f_dependent)))
        np.savetxt(pathdir_write_to+filename,dt)          

        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "cos")                
                
    except:
        return PA

    return PA


def get_exp(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=4):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
            variables = np.column_stack((variables,v))
        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))

        dt = np.column_stack((variables,np.exp(f_dependent)))
        np.savetxt(pathdir_write_to+filename,dt)       

        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "exp")                
                
    except:
        return PA

    return PA


def get_inverse(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=4):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
            variables = np.column_stack((variables,v))
        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))

        dt = np.column_stack((variables,1/f_dependent))
        np.savetxt(pathdir_write_to+filename,dt)          

        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "inverse")                    
                    
    except:
        return PA

    return PA


def get_log(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=4):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
            variables = np.column_stack((variables,v))
        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))

        dt = np.column_stack((variables,np.log(f_dependent)))
        np.savetxt(pathdir_write_to+filename,dt)         

        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "log")                    
                    
    except:
        return PA

    return PA


def get_sin(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=4):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
            variables = np.column_stack((variables,v))
        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))

        dt = np.column_stack((variables,np.sin(f_dependent)))
        np.savetxt(pathdir_write_to+filename,dt)         

        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "sin")                
                
    except:
        return PA

    return PA


def get_sqrt(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=4):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
            variables = np.column_stack((variables,v))
        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))

        dt = np.column_stack((variables,np.sqrt(f_dependent)))
        np.savetxt(pathdir_write_to+filename,dt) 

        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "sqrt")                    
                    
    except:
        return PA

    return PA


def get_squared(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=4):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
            variables = np.column_stack((variables,v))
        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))
   
        dt = np.column_stack((variables,f_dependent**2))
        np.savetxt(pathdir_write_to+filename,dt)

        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "squared")  
                
    except:
        return PA

    return PA


def get_tan(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=4):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
            variables = np.column_stack((variables,v))
        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))

        dt = np.column_stack((variables,np.tan(f_dependent)))
        np.savetxt(pathdir_write_to+filename,dt)        
        
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "tan")                
                
    except:
        return PA

    return PA









