import sympy
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
# from numpy.random import default_rng

#### Wrapper functions ####
def init_general_test_point(eq, X, y, bitmask):
    parsed_eq = parse_expr(eq)
    symbols = parsed_eq.free_symbols
    symbol_list = [x.name for x in symbols]
    symbol_list.sort(key=lambda x: int(x[1:]))
    return TestPoints(eq, symbol_list, X, y, bitmask)

def get_test_point(obj, full_pt):
    ''' Takes a N-dim vector, return N-dim vector '''
    pt = full_pt[obj.bitmask]
    opt = obj.find_reference_point(pt)
    in_pt = np.copy(full_pt)
    in_pt[obj.bitmask] = opt
    return in_pt


#### UTILS ####
# rng = default_rng()
def relative_error(hypot, bench):
    return np.abs((hypot-bench)/bench)

def project_plane(normal, point, guess):
    normal /= np.linalg.norm(normal)
    return guess - np.dot(guess-point, normal)*normal

class TestPoints:
    def __init__(self, eq, symbols, X, y, bitmask, mode='general'):
        '''
        mode is in {'general', 'add', 'minus', 'times', 'divide'} 
        eq, symbols are None if mode is not general
        '''
        self.mode = mode
        if mode == 'general':
            self.general_init(eq, symbols, X, y, bitmask)
        else:
            eq_map = {
                'add': 'x+y',
                'minus': 'x-y',
                'times': 'x*y',
                'divide': 'x/y',
            }
            if mode in eq_map:
                self.general_init(eq_map[mode], ['x', 'y'], X, y, bitmask)
            else:
                raise Exception("Unknown mode "+mode)
    def general_init(self, eq, symbols, X, y, bitmask):
        self.eq = eq
        self.symp_eq = sympy.sympify(eq)
        self.symbols = [sympy.symbols(x) for x in symbols]
        self.X = X
        self.y = y
        self.bitmask = bitmask
        self.symp_grads = [sympy.diff(self.symp_eq, x) for x in symbols]
        self.lambda_eq = sympy.lambdify(self.symbols, self.symp_eq)
        self.lambda_grads = [sympy.lambdify(self.symbols, x) for x in self.symp_grads]
        self.low_range = np.percentile(self.X[:, self.bitmask], 0, axis=0)
        self.high_range = np.percentile(self.X[:, self.bitmask], 100, axis=0)
        
        self.init_median_projection()
        self.init_scatter()

        self.log = []

    def init_median_projection(self):
        self.median_point = np.median(self.X[:, self.bitmask], axis=0)

    def init_scatter(self):
        self.hval = self.lambda_eq(*[self.X[:, i] for i in range(np.sum(self.bitmask))])
        self.hindices = np.argsort(self.hval)

    def evaluate_gradients(self, pt):
        r = np.array([f(*pt) for f in self.lambda_grads]).astype(float)
        return r

    def find_initial_guess_median_projection(self, pt):
        num_grads = self.evaluate_gradients(pt)
        num_grads /= np.linalg.norm(num_grads)
        return [project_plane(num_grads, pt, self.median_point)]
    

    def find_initial_guess_scatter(self, pt, low=2, high=3):
        guess = self.find_initial_guess_median_projection(pt)
        target_hval = self.lambda_eq(*pt)
        target_index = np.searchsorted(self.hval, target_hval, sorter=self.hindices)
        candidates = list(range(max(0, target_index-low), min(self.X.shape[0], target_index+high)))
        results = [self.X[self.hindices[guess], self.bitmask] for guess in candidates]
        return results
    
    def find_initial_guess_random(self, pt, num=2):
        return [self.X[t, self.bitmask] for t in np.random.randint(0, self.X.shape[0], num)]

    def optimize_fmin(self, guess, target):
        FTOL = 1e-4
        MAXITER = 200
        result = scipy.optimize.fmin(
            lambda x: np.abs(self.lambda_eq(*x) - target),
            guess, maxiter=MAXITER, ftol=FTOL, full_output=True, disp=False)
        if result[4] == 0:
            return result[0]
        else:
            return None
    def optimize_bfgs(self, guess, target):
        MAXITER = 20
        res = scipy.optimize.minimize(lambda x: .5 * (self.lambda_eq(*x) - target)**2,
                                    guess, method='BFGS', 
                                    jac=lambda x: self.evaluate_gradients(x) * (self.lambda_eq(*x) - target), 
                                    options={'maxiter':MAXITER, 'disp': False})
        if res.success:
            return res.x
        return None
    def optimize_basic(self, mode, guess, target):
        if mode == 'add':
            return project_plane(np.array([1., 1.]), np.array([target, 0]), guess)
        elif mode == 'minus':
            return project_plane(np.array([1., -1.]), np.array([target, 0]), guess)
        elif mode == 'divide':
            return project_plane(np.array([1., -target]), np.array([0, 0]), guess)
        elif mode == 'times':
            a, b = guess
            if target == 0:
                return np.array([a, 0]) if abs(a) >= abs(b) else np.array([0, b])
            else:
                A, B, C  = a, b**2 - a**2, -b*target
                x = (-B + (1 if target > 0 else -1) * (B**2 - 4*A*C)**0.5)/2/A
                return np.array([x, target/x])

    def in_domain(self, pt):
        return np.all(self.low_range <= pt) and np.all(pt <= self.high_range)
    def find_reference_point(self, pt):
        guesses = self.find_initial_guess_scatter(pt) + self.find_initial_guess_median_projection(pt) + self.find_initial_guess_random(pt)
        target_hval = self.lambda_eq(*pt)
        results = []
        for guess in guesses:
            if self.mode == 'general':
                result = self.optimize_bfgs(guess, target_hval)
            else:
                result = self.optimize_basic(self.mode, guess, target_hval)
            if result is not None and self.in_domain(result):
                results.append(result)
        return max(results, key=lambda x: np.linalg.norm(pt - x), default=None)
        
    
    def analyze_reference_point(self, pt, opt, disp):
        reference_point_rel_error = relative_error(self.lambda_eq(*opt), self.lambda_eq(*pt))
        reference_point_distance = np.linalg.norm(opt - pt)
        max_distance = np.linalg.norm(self.high_range - self.low_range)
        reference_point_rel_distance = reference_point_distance / max_distance
        self.log.append((reference_point_rel_error, reference_point_rel_distance))
#        if disp:
#            print(f'{pt} : found {opt}, err: {reference_point_rel_error}, distance: {reference_point_rel_distance}')

    def score_pt(self, model, full_pt, disp=False, log=False):
        pt = full_pt[self.bitmask]
        opt = self.find_reference_point(pt)
        if opt is None:
            return None
        if log:
            self.analyze_reference_point(pt, opt, disp)
        in_pt = np.copy(full_pt)
        in_pt[self.bitmask] = opt
        return relative_error(model(in_pt[np.newaxis, :]), model(full_pt[np.newaxis, :]))
