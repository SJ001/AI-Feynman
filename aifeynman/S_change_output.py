import numpy as np
import os
from .S_run_bf_polyfit import run_bf_polyfit

def get_acos(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=3):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    data = np.loadtxt(pathdir+filename)
    try:
        data[:,-1] = np.arccos(data[:,-1])
        np.savetxt(pathdir_write_to+filename,data)
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "acos")
    except:
        return PA

    return PA

def get_asin(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=3):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    data = np.loadtxt(pathdir+filename)
    try:
        data[:,-1] = np.arcsin(data[:,-1])
        np.savetxt(pathdir_write_to+filename,data)
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "asin")
    except:
        return PA

    return PA

def get_atan(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=3):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    data = np.loadtxt(pathdir+filename)
    try:
        data[:,-1] = np.arctan(data[:,-1])
        np.savetxt(pathdir_write_to+filename,data)
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "atan")
    except:
        return PA

    return PA


def get_cos(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=3):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    data = np.loadtxt(pathdir+filename)
    try:
        data[:,-1] = np.cos(data[:,-1])
        np.savetxt(pathdir_write_to+filename,data)
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "cos")
    except:
        return PA

    return PA


def get_exp(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=3):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    data = np.loadtxt(pathdir+filename)
    try:
        data[:,-1] = np.exp(data[:,-1])
        np.savetxt(pathdir_write_to+filename,data)
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "exp")
    except:
        return PA

    return PA


def get_inverse(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=3):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    data = np.loadtxt(pathdir+filename)
    try:
        data[:,-1] = 1/data[:,-1]
        np.savetxt(pathdir_write_to+filename,data)
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "inverse")
    except:
        return PA

    return PA


def get_log(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=3):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    data = np.loadtxt(pathdir+filename)
    try:
        data[:,-1] = np.log(data[:,-1])
        np.savetxt(pathdir_write_to+filename,data)
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "log")
    except:
        return PA

    return PA


def get_sin(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=3):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    data = np.loadtxt(pathdir+filename)
    try:
        data[:,-1] = np.sin(data[:,-1])
        np.savetxt(pathdir_write_to+filename,data)
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "sin")
    except:
        return PA

    return PA


def get_sqrt(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=3):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    data = np.loadtxt(pathdir+filename)
    try:
        data[:,-1] = np.sqrt(data[:,-1])
        np.savetxt(pathdir_write_to+filename,data)
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "sqrt")

    except:
        return PA

    return PA


def get_squared(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=3):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    data = np.loadtxt(pathdir+filename)
    try:
        data[:,-1] = data[:,-1]**2
        np.savetxt(pathdir_write_to+filename,data)
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "squared")

    except:
        return PA

    return PA


def get_tan(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg=3):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    data = np.loadtxt(pathdir+filename)
    try:
        data[:,-1] = np.tan(data[:,-1])
        np.savetxt(pathdir_write_to+filename,data)
        PA = run_bf_polyfit(pathdir,pathdir_write_to,filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg, "tan")

    except:
        return PA

    return PA



