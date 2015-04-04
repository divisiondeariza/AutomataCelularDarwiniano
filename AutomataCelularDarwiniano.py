# -*- coding: utf-8 -*-
"""


"""
import numpy as np
import matplotlib.pylab as plt
import time
from matplotlib import animation
from random import shuffle

"""
This function generates a random square matrix of pixels of the given size and with a probability p for any pixel of being "inhabited".
size: size of the square matrix
p: probability for any pixel being "inhabited" 

RETURNS
M: pixel matrix inicialized
H: set which contains the coordinates for all vectors inicialized
"""
def initialMatrixGenerator(size=10, p=0.5):
    M = np.zeros([size, size, 3],dtype=np.uint8)
    H = set([])

    for i in range(size):
        for j in range(size):
            [M[i,j,:], mutated] = mutatePixel(M[i,j,:], p)
            if mutated: H.add((i,j))
            
            
    return [M,H]

"""
This function returns a pixel which have a probability prob of suffering a transformation
pixel: pixel which have a probability prob of being transformed
prob: probability of the output pixel for being diferent (or mutated) prom the input one

RETURNS:
Output_Pixel = pixel with mutation rule applied
mutated = boolean which indicates if the pixel was modified
"""
def mutatePixel(pixel, prob):
    """
    p_singleval is tre probability for each color of a pixel for being changed in any pixel
    If we consider the probability for all three components for being inicialized with a non zero value and p being the probability for a pixel being diferent to [0,0,0] we have that:
    p = 3*p_singleval - 3*p_singleval^2 + p_singleval^3     if we consider the change of any component in a pixel as an independent event. 
    
    Thus, solving we get only one real solution:
    p_singleval = 1 - (1-p)^(1/3)
    """
    p_singleval = 1 - np.power(1-prob,1/3.0) 
    #print p_singleval
    mutated = False
    nPixel = pixel
    for k in range(3):
        if np.random.random() < p_singleval:
            nPixel[k]=np.random.random_integers(255)
            mutated=True
                
    return [nPixel, mutated]

"""
This fuction returns the "survival value" which is the value that will be used for determine which pixel will hold a position in the array. For avoid an absoulute advantage for any kind pixel in any condition, any pixel chosen randomly should have the same probabity for having any of the possible survival value, in this case is chosen the index of the component whom value is the maximum of the difference between the pixel and the vector-shifted-pixel. Thus for an arbitrary pixel the probability for getting any of the three posible values {0,1,2} is almost the same. (actually is about 0.6% less likely getting a 2 than getting a 1 or a 0).
Seems weird to take the index for the max of the diferences between components instead of just taking the max for the pixel-vector. The reason is that is needed to have al components interrelated for avoid long-term advantages.

pixel: the pixel from which will bw obtained the Survival Value
s_value: The Survival value itself

RETURNS The survival value
"""
def survivalValue(pixel):
    return np.argmax(pixel - np.roll(pixel,1))
    
"""
This function compares two pixels in a circular way "Rock-paper-scissors style" using their survival values. In order to guarantee a advantadge/ disadvantage for all pixels only dependent of the current status of the neighbors and avoid an absolute advantage/disadvantage (I.E. A pixew which is broadly more likely to win/lose than the others) the comparison must accomplish two conditions:
1- all the possible Survival Values are equaly distributed over the space of all the possoble pixels.
2- every survival value wins with the half of possible survival differents to itself values and loses with the other half. 
3- The probability for a pixel who currently inhabites a cell with a new 'invasor' pixel defines the stability for the sequences.

pixel_a: The pixel who inhabites a cell
pixel_b: The pixel who invades the cell
p_stability: probability for a inhabitant pixel to win against the invasor pixel

RETURNS

"""
def comparePixels(pixel_a,pixel_b, p_stability):
    a = survivalValue(pixel_a)
    b = survivalValue(pixel_b)
    
    #If sNumber<0 survives a, if sNumber>0 survives b, else they get tied (are the same) and are defined for the p_stability probability
    sNumber = ((a - b)%3 -1.5)*(a!=b)
    
    if(sNumber == 0):
        if np.random.random() < p_stability:
            return pixel_a
        else:
            return pixel_b
    elif(sNumber<0):
        return pixel_a
    else: 
        return pixel_b


def generateNextMatrix(data,M,H):
    size = M.shape[0]
    H_new = H.copy()
    M_new = M
    for (i,j) in H:
        h_cells = habitableCells(i, j, size, p=0.2)
        for (i_child,j_child) in h_cells:
            #print i_child
            child_pixel, mutate = mutatePixel(M[i,j,:], 0.001)
            if (i,j) in H_new:
                M_new[i_child,j_child,:] = comparePixels(child_pixel,M_new[i,j,:], 0.5)
            else:
                M_new[i_child,j_child,:] = child_pixel
                
            H_new.add((i_child,j_child))
        pass
    
#        i_new = (i+1)%size
#        j_new = (j-1)%size
#        M_new[i_new, j_new,:],mutate = 
#        #M_new[j,i,:]=M[i,j,:]
#        
    
        
    return [M_new, H_new]
    
"""
Returns the cells that a pixel should populate following a probabilistic rule (binomial), a pixel only can reproduce on the adjacent cells, thus the pixel can reproduce himself 8 times per cycle at most.

x: x coordinate od the pixel
y: y coordinate od the pixel
size: size of the square array
p: probability of success for a pixel to reproduce himself on an adjacent cell

"""
def habitableCells(xcord, ycord, size, p=0.5):
    #Set of all the adjacent cells for the pixel
    adj_set=set([]);
    for i in range(-1,2):
        for j in range(-1,2):
            adj_set.add(( (xcord+i)%size, (ycord+j)%size ))
            
    #Removes the cell where the father pixel lives.        
    adj_set.remove((xcord,ycord)) 
    
    #Number of childs  
    N = np.random.binomial(8, p)    
    adj_list = list(adj_set)
    shuffle(adj_list)
    return adj_list[:N]
    
"""
Finally, this function calls the function whom iterates over the current matrix and returns a plotable entity.
 
"""    
def mSlideShow(data):
    global M,H
    [M, H] = generateNextMatrix(data,M,H)
    mat.set_data(M)
    return [mat]
    pass
    

"""
A test for probability consistence
"""    
def testSurvivalValueDist(M,H,p):
    s_pixels = []
    s_wins= []
    for (i,j) in H:
        s_pixels.append(M[i,j,:])
    
    for i in range(90000):
        a = s_pixels[np.random.random_integers(len(s_pixels)-1)]
        b = s_pixels[np.random.random_integers(len(s_pixels)-1)]
        winner=comparePixels(a,b,p);
        s_wins.append((survivalValue(winner) + 1)*np.power(-1,sum(winner==b)/3))
    plt.hist(s_wins,bins=7)
    #plt.hist([survivalValue(ix) for ix in s_pixels],bins=3)
    plt.show()
    
"""
Routine 
"""   
[M, H] = initialMatrixGenerator(20,0.05)
plt.rc('axes',edgecolor='green')
# set up animation
fig, ax = plt.subplots()
ax.set_axis_bgcolor('red')
mat = ax.imshow(M, interpolation='spline36', aspect='normal')
ani = animation.FuncAnimation(fig, mSlideShow, interval=100, frames=500)
plt.axis('off')
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
#ani.save('TSD.mp4', metadata={'artist':'Emmanuel Ariza'}, extra_args=['-vcodec', 'libx264'])
plt.show()


            
            
            
            
            
