from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedKeyList


class Point(object):
    def __init__(self, x, y, data=None, id=None):
        self.x = x
        self.y = y
        self.data = data
        self.id = id


    def __getitem__(self, index):
        """Indexing: get item according to index."""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.data
        elif index == 3:
            return self.id
        else:
            raise Exception("Index {} is out of range!".format(index))


    def __setitem__(self, index, value):
        """Indexing: set item according to index."""
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.data = value
        elif index == 3:
            raise Exception("Cannot set Id!")
        else:
            raise Exception("Index {} is out of range!".format(index))


# In[2]:


class ParetoSet(SortedKeyList):
    """Maintained maximal set with efficient insertion. Note that we use the convention of smaller the better."""

    def __init__(self):
        super().__init__(key=lambda p: p.x)


    def _input_check(self, p):
        """Check that input is in the correct format.

        Args:
            p: input

        Returns:
            Point:

        Raises:
            TypeError if cannot be converted.
        """

        if isinstance(p, Point):
            return p
        elif isinstance(p, tuple) and len(p) == 2:
            return Point(x=p[0], y=p[1], data=None)
        else:
            raise TypeError("Must be instance of Point or 2-tuple.")
    
    
    def get_id_list(self):
        id_list = []
        for point in self:
            id_list.append(point.id)
        return id_list

    
    def add(self, p):
        """Insert Point into set if minimal in first two indices.

        Args:
            p (Point): Point to insert

        Returns:
            bool: True only if point is inserted

        """
        p = self._input_check(p)

        is_pareto = False
        # check right for dominated points:
        right = self.bisect_left(p)

        while len(self) > right and self[right].y >= p.y and not (self[right].x == p.x and self[right].y == p.y):
            self.pop(right)
            is_pareto = True

        # check left for dominating points:
        left = self.bisect_right(p) - 1

        if left == -1 or self[left][1] > p[1]:
            is_pareto = True

        # if it's the only point it's maximal
        if len(self) == 0:
            is_pareto = True

        if is_pareto:
            super().add(p)

        return is_pareto


    def __contains__(self, p):
        p = self._input_check(p)

        left = self.bisect_left(p)

        while len(self) > left and self[left].x == p.x:
            if self[left].y == p.y:
                return True

            left += 1

        return False


    def __add__(self, other):
        """Merge another pareto set into self.

        Args:
            other (ParetoSet): set to merge into self

        Returns:
            ParetoSet: self

        """

        for item in other:
            self.add(item)

        return self


    def distance(self, p):
        """Given a Point, calculate the minimum Euclidean distance to pareto
        frontier (in first two indices).

        Args:
            p (Point): point

        Returns:
            float: minimum Euclidean distance to pareto frontier

        """
        p = self._input_check(p)

        point = np.array((p.x, p.y))
        dom = self.dominant_array(p)

        # distance is zero if pareto optimal
        if dom.shape[0] == 0:
            return 0.

        # add corners of all adjacent pairs
        candidates = np.zeros((dom.shape[0] + 1, 2))
        for i in range(dom.shape[0] - 1):
            candidates[i, :] = np.max(dom[[i, i+1], :], axis=0)

        # add top and right bounds
        candidates[-1, :] = (p.x, np.min(dom[:, 1]))
        candidates[-2, :] = (np.min(dom[:, 0]), p.y)

        return np.min(np.sqrt(np.sum(np.square(candidates - point), axis=1)))


    def dominant_array(self, p):
        """Given a Point, return the set of dominating points in the set (in
        the first two indices).

        Args:
            p (Point): point

        Returns:
            numpy.ndarray: array of dominating points

        """
        p = self._input_check(p)

        idx = self.bisect_left(p) - 1

        domlist = []

        while idx >= 0 and self[idx][1] < p[1]:
            domlist.append(self[idx])
            idx -= 1

        return np.array([x[0:2] for x in domlist])


    def to_array(self):
        """Convert first two indices to numpy.ndarray

        Args:
            None

        Returns:
            numpy.ndarray: array of shape (len(self), 2)

        """
        A = np.zeros((len(self), 2))
        for i, p in enumerate(self):
            A[i, :] = p.x, p.y

        return A

    def get_pareto_points(self):
        """Returns the x, y and data for each point in the pareto frontier
        
        """
        pareto_points = []
        for i, p in enumerate(self):
            pareto_points = pareto_points + [[p.x, p.y, p.data]]
        
        return pareto_points
        

    def from_list(self, A):
        """Convert iterable of Points into ParetoSet.

        Args:
            A (iterator): iterator of Points

        Returns:
            None

        """
        for a in A:
            self.add(a)
    
    
    def plot(self):
        """Plotting the Pareto frontier."""
        array = self.to_array()
        plt.figure(figsize=(8, 6))
        plt.plot(array[:, 0], array[:, 1], 'r.')
        plt.show()


if __name__ == "__main__":
    PA = ParetoSet()
    A = np.zeros((40, 2))
    
    for i in range(40):
        x = np.random.rand()
        y = np.random.rand()
        
        A[i, 0] = x
        A[i, 1] = y
        
        PA.add(Point(x=x, y=y, data=None))
    paretoA = PA.to_array()

