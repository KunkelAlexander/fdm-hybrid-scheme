import matplotlib.pyplot as plt 
import numpy as np 

import matplotlib.patches as patches

wave_threshold = 0.0
splitting_threshold = 0.8
density_threshold = 1e-10
dsi_threshold = 20
interval_length_threshold = 16

class Node:
   def __init__(self, level, N, N0, boxWidth, dimension, createCallback):
      self.level          = level
      self.children       = []
      self.N0             = N0
      self.boxWidth       = boxWidth
      self.N              = N
      self.dim            = dimension
      self.callback       = None
      self.createCallback = createCallback 
      if dimension < 3:
         self.minlevel = 1
      else:
         self.minlevel = 2


   def prange(self, low, high):
      if (low > high):
         low -= self.N
      return np.array(range(low, high)) % self.N

   def checkSplit(self, fields):
      if self.dim == 1:
         subx    = self.prange(self.N0, self.N0 + self.boxWidth)
         density = fields[0][subx]
         dsi     = fields[1][subx]
      elif self.dim == 2:
         suby    = self.prange(self.N0[0], self.N0[0] + self.boxWidth).reshape(-1, 1)
         subx    = self.prange(self.N0[1], self.N0[1] + self.boxWidth).reshape( 1,-1)
         density = fields[0][suby, subx]
         dsi     = fields[1][suby, subx]

      cond = np.logical_or((density < density_threshold),(dsi > dsi_threshold))
      return np.sum(cond)/self.boxWidth**self.dim

   def update(self, fields):
      #Check splitting condition
      ratio = self.checkSplit(fields)

      #If the level interference is below the threshold, we do not need subdivision or wave solver
      if (ratio <= wave_threshold and self.level >= self.minlevel):
         self.children  = []
         self.callback  = None
         return 

      #Else proceed to turn on wave solver and subdivide node
      half_length  = int(self.boxWidth/2)

      #Do not subdivide if resulting nodes are too small or most of the node requires wave solver
      if ((half_length < interval_length_threshold) or (ratio >= splitting_threshold)) and self.level >= self.minlevel:
         self.children  = []
         if self.callback is None and self.createCallback is not None:
            self.callback = self.createCallback(self)
         return 

      #If the node requires splitting and does not already have children, add children
      if not self.children:
         N0 = self.N0
         boxWidth  = half_length + 2
         if self.dim == 1:
            n01 = N0 - 1
            n02 = N0 - 1 + half_length 
            self.children.append(Node(self.level + 1, self.N, n01, boxWidth, self.dim, self.createCallback))
            self.children.append(Node(self.level + 1, self.N, n02, boxWidth, self.dim, self.createCallback))
         elif self.dim == 2:
            n01 = np.array([N0[0] - 1              , N0[1] - 1])
            n02 = np.array([N0[0] - 1              , N0[1] - 1 + half_length])
            n03 = np.array([N0[0] - 1 + half_length, N0[1] - 1])
            n04 = np.array([N0[0] - 1 + half_length, N0[1] - 1 + half_length])
            self.children.append(Node(self.level + 1, self.N, n01, boxWidth, self.dim, self.createCallback))
            self.children.append(Node(self.level + 1, self.N, n02, boxWidth, self.dim, self.createCallback))
            self.children.append(Node(self.level + 1, self.N, n03, boxWidth, self.dim, self.createCallback))
            self.children.append(Node(self.level + 1, self.N, n04, boxWidth, self.dim, self.createCallback))
         else:
            raise ValueError()

      #Update children
      for child in self.children:
         child.update(fields)


   def getCallbacks(self, callbacks = []):
      if self.callback is not None:
         callbacks.append(self.callback)
         
      for child in self.children:
         child.getCallbacks(callbacks)

# Print the Tree
   def plotTree(self, ax, root = False):
      if root:
         plt.title(f"Binary tree in {self.dim}D")

         if self.dim == 1:
            plt.xlabel("N")
            plt.ylabel("# levels")
         elif self.dim == 2:
            ax.set_aspect("equal")
            ax.set_ylim([-10, self.boxWidth + 10])
            ax.set_xlim([-10, self.boxWidth + 10])

      for child in self.children:
         child.plotTree(ax)

      if self.dim == 1:
         xx = np.arange(self.N0, self.N0 + self.boxWidth)
         if (self.callback is not None):
            ax.scatter(xx, self.level * np.ones(xx.shape[0]), c = "r", alpha = 0.3)
         else:
            ax.scatter(xx, self.level * np.ones(xx.shape[0]), c = "k", alpha = 0.3)

      elif self.dim == 2 and len(self.children) == 0: 
         if (self.callback is not None):
            c = "r"
         else:
            c = "k"

         ax.add_patch(patches.Rectangle(self.N0, self.boxWidth, self.boxWidth,
                     edgecolor = c,
                     facecolor = c,
                     fill=True,
                     alpha = 0.2))
