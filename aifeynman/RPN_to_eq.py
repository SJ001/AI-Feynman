# Turns an RPN expression to normal mathematical notation

import numpy as np

def RPN_to_eq(expr):

    variables = ["0","1","a","b","c","d","e","f","g","h","i","j","k","l","m","n","P"]
    operations_1 = [">","<","~","\\","L","E","S","C","A","N","T","R","O","J"]
    operations_2 = ["+","*","-","/"]

    stack = np.array([])

    for i in (expr):
        if i in variables:
            if i == "P":
                stack = np.append(stack,"pi")
            elif i == "0":
                stack = np.append(stack,"0")
            elif i == "1":
                stack = np.append(stack,"1")  
            else:
                stack = np.append(stack,"x" + str(ord(i)-97))
        elif i in operations_2:
            a1 = stack[-1]
            a2 = stack[-2]
            stack = np.delete(stack,-1)
            stack = np.delete(stack,-1)
            a = "("+a2+i+a1+")"
            stack = np.append(stack,a)
        elif i in operations_1:
            a = stack[-1]
            stack = np.delete(stack,-1)
            if i==">":
                a="("+a+"+1)"
                stack = np.append(stack,a)
            if i=="<":
                a="("+a+"-1)"
                stack = np.append(stack,a)
            if i=="~":
                a="(-"+a+")"
                stack = np.append(stack,a)
            if i=="\\":
                a="("+a+")**(-1)"
                stack = np.append(stack,a)
            if i=="L":
                a="log("+a+")"
                stack = np.append(stack,a)
            if i=="E":
                a="exp("+a+")"
                stack = np.append(stack,a)
            if i=="S":
                a="sin("+a+")"
                stack = np.append(stack,a)
            if i=="C":
                a="cos("+a+")"
                stack = np.append(stack,a)
            if i=="A":
                a="abs("+a+")"
                stack = np.append(stack,a)
            if i=="N":
                a="asin("+a+")"
                stack = np.append(stack,a)
            if i=="T":
                a="atan("+a+")"
                stack = np.append(stack,a)
            if i=="R":
                a="sqrt("+a+")"
                stack = np.append(stack,a)
            if i=="O":
                a="(2*("+a+"))"
                stack = np.append(stack,a)
            if i=="J":
                a="(2*("+a+")+1)"
                stack = np.append(stack,a)
    return(stack[0])
