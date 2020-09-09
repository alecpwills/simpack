"""
(simpack.)classes.py
A variety of classes to analyze molecular dynamics trajectories

Handles the primary functions
"""
from abc import ABC, abstractmethod
import os
import pickle

#TODO: Update base simulation to be more consistent with the sbu-ccmp-tools one I did
#TODO: ALSO: Rewrite entire SiestaSimulation to not use pandas, move to numpy

class Simulation(ABC):
    @abstractmethod
    def __init__(self):
        self.path = ''
    
    @abstractmethod
    def itrajectory():
        pass
    
    def save(self, overwrite = True):
        pklp = os.path.join(self.path, 'self.pkl')
        #if file exists, assert we want to overwrite
        if os.path.exists(pklp):
            assert overwrite, "A saved instance of this object already exists"
        print("Saving: {}".format(pklp))
        save_dict = vars(self)
        keys = list(save_dict.keys())
        with open(pklp, 'wb') as f:
            pickle.dump((keys, save_dict), f)

    def load(self):
        with open(os.path.join(self.path, 'self.pkl'), 'rb') as f:
            keys, load_dict = pickle.load(f)
            for k in keys:
                setattr(self, k, load_dict[k])



def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote




if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
