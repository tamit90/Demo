#!/usr/bin/env python
# coding: utf-8

# # Introduction to Data Science and Systems 2020-2021<small><small>v20202021a</small></small>
# ## Lab 1: Computational Linear Algebra
# #### - ***you should submit the completed notebook to Moodle along with one pdf file (see the end of the notebook and Moodle for instructions)***
# 
# #### University of Glasgow, JHW & BSJ, 2020-2021

# ## Guide
# 
# Lab 1 is structured as follows (with two main task sections):
# 
# >-    **Intro: A whisky dataset** 
# >-    **Task A: Norms, interpolation and statistics**
# >-    **Task B: Eigenvectors and PCA**
# >-    **Task C: System of linear equations**
# >-    **Appendix: Marking Summary (and additional metadata)**
# 
# You will need to understand the following functions well to complete this lab:
# * [`np.argmin()`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.argmin.html) [Self-study]
# * [`np.argsort()`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.argsort.html) [Self-study]
# * [`np.linalg.norm()`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.norm.html) [Week 2]
# * [`np.linalg.svd()`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.svd.html) [Week 3]
# * [`np.linalg.eig()`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.eig.html) [Week 3]
# * [`np.cov()`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.cov.html) [Week 2]
# * [`np.linalg.pinv()`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.pinv.html) [Week 3]
# 
# 
# We recommend you read through the lab *carefully* and work through the tasks one by one.
# 
# #### Material and resources 
# - It is recommended to keep the lecture notes (from lecture 1-2) open while doing this lab exercise. 
# - If you are stuck, the following resources are very helpful:
#  * [NumPy cheatsheet](https://github.com/juliangaal/python-cheat-sheet/blob/master/NumPy/NumPy.md)
#  * [NumPy API reference](https://docs.scipy.org/doc/numpy-1.13.0/reference/)
#  * [NumPy user guide](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.html)
# 
# #### Marking and Feedback
# This assessed lab is marked using two different techniques;
# 
# - Autograded with feedback; you'll get immediate feedback.
# - Autograded without (immediate) feedback (there will always be a small demo/test so you can be confident that the format of your answer is correct).
# 
# *Note*: auto-graded results are always provisional and subject to change in case there are significant issues (this will be in favor of the student).
# 
# #### Help \& Assistance
# - This lab is graded and the lab assistants/lecturer can provide guidance but we can (and will) not give you the final answer or confirm your result.
# 
# #### Plagiarism
# - All submissions will be automatically compared against each other so make sure your submission represents an independent piece of work! We have provided a few checks to make sure that is indeed the case.
# 

# ---

# # Before you begin
# 
# Please update the tools we use for the automated greading by running the below command (uncomment) and restart your kernel (and then uncomment again) -- or simply perform the installation externally in an Anaconda/Python prompt.

# In[3]:


# !pip install -U --force-reinstall --no-cache https://github.com/johnhw/jhwutils/zipball/master


# Let's import some useful Python packages and define a few custom functions...

# In[4]:


# Standard imports
# Make sure you run this cell!
from __future__ import print_function, division
import numpy as np  # NumPy
import scipy.stats 
import os
import sys
import binascii
from unittest.mock import patch
from uuid import getnode as get_mac

from jhwutils.checkarr import array_hash, check_hash, check_scalar, check_string
import jhwutils.image_audio as ia
import jhwutils.tick as tick

###
tick.reset_marks()

# special hash funciton
def case_crc(s, verbose=True):
    h_crc = binascii.crc32(bytes(s.lower(), 'ascii'))
    if verbose:
        print(h_crc)
    return h_crc

# this command generaties a unique key for your system/computer/account
uuid_simple = (("%s") % get_mac())
uuid_str = ("%s\n%s\n%s\n%s\n%s\n") % (os.path,sys.path,sys.version,sys.version_info,get_mac())
uuid_system = case_crc(uuid_str,verbose=False) 


# Set up Matplotlib
import matplotlib as mpl   
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc('figure', figsize=(8.0, 4.0), dpi=140)
np.random.seed(2021)

# You can ignore this
import lzma, base64
exec(lzma.decompress(base64.b64decode(b'/Td6WFoAAATm1rRGAgAhARYAAAB0L+Wj4AuQBIRdADSbSme4Ujxz95HdWLf4m39SX9b5yuqRd8UVk3DwULgMdEb8P1bLvis2Swz3hlDU+FwGvQQSXUZEv0GMy+ErARv0E4TRCmTvFyQ5JSb/G7uf1mJI1cRyyqS/p8OjxfizueERZipJkqifEM7mgPLl2o+B4OX/p+0Vu3LfTMRZvY//6V0JXZwRxDGVuddVdlsZOuNDNEzsXiiyo2fiXL31w0sXabfigUkJ4q1uo4W7C0THX7Lhb0zVk9A0/+f114CeChR9Gz25xOstbGjOodl1SDpP5rKIxkxZTzcjw65yJRKqid46CTa5ffWK7y9QtygL7IqcGB9ode3TKcTh57Edd9+wylW9BiSE8/qh/93qFavlXK1sjLdoTWEfPZ96HOR6La9g8KEEFNNMAc+7HtTv1JwoX8w+zLayLzdpIqA+HLAVFiVeFg4jPu8imDtmoOhe66WDgPWsXetR6FFCK7mw4q4Q+A4TCz6ugUoh/gIEGjIadianBQVSST062b8vSLzWmFYLKJoRPDHPyvFWKX5u27LZ9Bpswls2feqW6SBEvavwjckRwW5r4Fc6F+MEYgZmAUy8sJRXe7JHvp6LZ3o5RM92eQRoGsDeL82U2LC6sXfxy3MBj6Gd0wwWC9iJdyvs+laSdI41jk2lZUcDpVCBoSV/Zr+0rH1PsT33u2NlfDsaXrG67zKhbB+SSGz3OoN6Kq/1GwWf+GvNH3cySyrJOgN2edwh/fn87XMHk5QZCg0BtZRObATtZAoloB5jJGvjwqtxHCItkTdGoUi4TY75N3FMTPowFYUXn2tjAtngJibqtGbZ/+PaS7E134Lsvxy5o2uaBgoV+U9Mg1poz1QAl0YTKDNMZjVILDbKIRq9e2C4X6e3SWQRW4LrBujBJp7Q8AJjKIspFOLt7PzxOwkSHES90iNgMW4Sn+uTKwQEcTrtTZCDm5Bynn5taepEXp2hj2cmuZEGJCXX8HOM9RgnWyeOVDcUPRCGAGrjA3y7VZGuEjdPE4DT32dmqJabHrPtrc0tgde5UfefS8ezzGOEheOmYQEtpIZLY2TwuNbhOIvIxNfnDA7H7ug1LtCSTejYkGU9CztGzKkyoWEMSTGQSd7aEddrdDS8gsOF6r+RmhCutjGejXFHFtVEcL8FJxczfLdbWDNdBl69IrZ8vlV6Ts4FojBO0/w6HAv24jyX1r+4n3ymPeJZb2SR7HQ/4L2In4ywuUdCkI2t2UuB0fHYgA+ibCVPoXg5Da698PlcozIlD/cmP+3OAnEU+yPElHmLrfjGLFwmWN28ikbluPx0be9B7sn4qTJUY0zrOBuv+wS47A7j5XXicpakCHJcqDaEuzWCa6e1JRmIDoitnr+2kNbGDYNPgKKJE8XDvWVZTgnG1NCGhTZJlTL37hZZIuwkA5RbpnOlrEldKjGnol9D209OuritES1GvlL2H7lDtRTiMnHPHcHMnVqPg5usk3F2Zw23PtC1YDaHvqxgyqaXlRslElFtLz2k9GV2QC3bUxVVlf6jQgPkDoQhKu63JjQtoPRrn0AAR37PsnsFZ74AAaAJkRcAAHaUzMuxxGf7AgAAAAAEWVo=')).decode('ascii'))
print("Everything imported OK (ignore deprecated warnings)")


# In[5]:


# Hidden cell for utils needed when grading (you can/should not edit this)


# ---
# 

# **Mini-task**: provide your personal details in two variables:
# 
# * `student_id` : a string containing your student id (e.g. "1234567x"), must be 8 chars long.
# * `student_typewritten_signature`: a string with your name (e.g. "Adam Smith") which serves as a declaration that this is your own work (read the declaration of originality when you submit on Moodle).*

# In[6]:


student_id = "2602824H" # your 8 char student id
student_typewritten_signature = "Tamit Halder" # your full name, avoid spceical chars if possible

# YOUR CODE HERE
#raise NotImplementedError()


# In[7]:


## We will print your info to a pdf file at the end of the notebook - 
# including the Declaration of Originality - which must be uploaded alongside 
# the actual notebook  you should also see two green checkmarks [0 marks] 
# indicating that your info meet the basic std)

with tick.marks(0): # you don't get any credit for remembering your student id. This is just a test!
    assert(len(student_id)==8)

with tick.marks(0):  # you don't get any credit for remembering your own name! This is just a test!
    assert(len(student_typewritten_signature)>0)


# ----

# ## Introduction
# ## Whisky: Representing and comparing vectors

# Whisky distillation is a major industry in Scotland. 
# 
# <img src="imgs/stills.jpg"> <br><br>*.[Image](https://flickr.com/photos/sashafatcat/518104633 "stills") by [sashafatcat](https://flickr.com/people/sashafatcat) shared [CC BY](https://creativecommons.org/licenses/by/2.0/)*
# 
# The dataset in `data/whisky.csv` is data from a number of whisky distilleries. For each distillery, there is a set of subjective judgements about the flavour characteristics of their product. The data comes from [this Strathclyde University research project](https://www.mathstat.strath.ac.uk/outreach/nessie/nessie_whisky.html).
# 
# Each distillery has been judged on twelve flavour indicators (like "smokiness" or "sweetness"), and they have been assigned values from 0-4, indicating the strength of that category as judged by an expert whisky drinker. These can be seen as 12D vectors, one vector per distillery. **Every distillery is represented as a point in twelve dimensional vector space.**
# 
# We also have a 2D array of the geographical locations of each distillery. The code below loads the data.
# 
# ## Loading the data

# In[8]:


## It is not necessary to understand this code to complete the exercise. 
import pandas as pd

whisky_df = pd.read_csv("data/whiskies.txt")
whisky_df = whisky_df.sort_values(by="Distillery")
# extract the column and row names
distilleries = np.array(whisky_df["Distillery"])

columns = {name.lower(): index for index, name in enumerate(whisky_df.columns[2:-3])}

# split apart the data frame and form numpy arrays
locations = np.array(whisky_df.iloc[:, -2:])
whisky = np.array(whisky_df.iloc[:, 2:-3])

# fix wine column which is misnamed
columns["wine"] = columns["winey"]
del columns["winey"]
# force tie breaks
np.random.seed(2018)
whisky = whisky + np.random.normal(0, 0.1, whisky.shape)


# ## Viewing the data

# We can see this whole dataset as a heatmap:

# In[9]:


# show a plot of the whisky data
fig = plt.figure(figsize=(10,25))
ax = fig.add_subplot(1,1,1)

# image plot
img = ax.imshow(whisky)
ax.set_yticks(np.arange(len(distilleries)))
ax.set_yticklabels(distilleries, rotation="horizontal", fontsize=12)

# put the x axis at the top
ax.xaxis.tick_top()
ax.set_xticks(np.arange(len(columns)))
ax.set_xticklabels(columns, rotation="vertical", fontsize=12)

# some horrific colorbar hackery to put in the right place
# don't worry about this bit!
cbaxes = fig.add_axes([0.37, 0.93, 0.28, 0.01])  
fig.colorbar(img, orientation='horizontal',  cax=cbaxes, ticks=np.arange(5))
cbaxes.xaxis.tick_top()


# ### Available data
# You now have these variables:
# 
# * `whisky` an  86x12 array of taste judgements, one row for each of the 86 distilleries. Each whisky has a rating 0-4 for each of the 12 flavour categories.
# * `distilleries` is a list of 86 distillery names
# * `columns` is a mapping of feature names to column indices.
# * `locations` is an 86x2 matrix of positions of each distillery in [OS grid reference format](https://www.gridreferencefinder.com/) in the same order as `whisky`
# 
# For example:

# In[10]:


print(whisky[distilleries.searchsorted('Glenfiddich'), 
             columns['smoky']])


# will tell you how "smoky" Glenfiddich was rated.

# In[11]:


print(distilleries[8]) # distilleries is just a list of names


# will tell you the 9th distillery in the dataset is `Aultmore`.

# In[12]:


print(locations[distilleries.searchsorted('Glengoyne')])


# will tell you where to find the Glengoyne distillery in UK OS grid units.

# ----

# ## Task A: Norms, interpolation and statistics
# We will see some simple things we can do with this dataset. 
# 

# We can compute distances in **flavour space** between distilleries.
# 
# For example, we can compute the distance between the `Glenlivet` distillery and every other distillery *in terms of flavour, not physical distance*. The result will be a 1D array of 86 distances. 
# 
# Remember: distance between $\vec{x}$ and $\vec{y}$ is the norm of their difference: $\|\vec{x}-\vec{y}\|$
# 
# We can compute this for several different norms (e.g. $L_1, L_2$, and $L_\infty$)
# 
# We start by subtracting the flavour vector for `Glenlivet` from all the other flavour vectors in the `whisky` matrix.
# 
# We then use `np.linalg.norm` to compute the norm of every row vector in the resulting matrix. We set `axis=1` to ensure that norms are calculated "across columns", i.e. the norm of each row vector is calculated. We also specify which norm we want to calculate (`1` for $L_1$, `2` for $L_2$ or `np.inf` for $L_\infty$). The result is a 1D array of 86 distances in **flavour space**.

# In[13]:


# Get the flavour vector for Glenlivet
glenlivet = whisky[distilleries.searchsorted("Glenlivet")]
print(glenlivet)


# In[14]:


## Compute distances
# must use axis=1 to get the right result, otherwise the matrix norm will be used
# (the matrix norm is calculated across the whole matrix, rather than across each row vector!)
glenlivet_1 = np.linalg.norm(whisky - glenlivet, 1, axis=1)  # L_1
glenlivet_2 = np.linalg.norm(whisky - glenlivet, 2, axis=1)  # L_2
glenlivet_inf = np.linalg.norm(whisky - glenlivet, np.inf, axis=1)  # L_inf


# ### Visualising these distances
# We can visualise these distances. This lets us see which distilleries produce whisky most similar to Glenlivet and which of them are most dissimilar. Note that we can use `argsort` to order a list of distances. Below, we plot a faceted graph, one facet for each norm, showing each distillery's flavour distance to `Glenlivet` as a rank bar plot (see Unit 3). The labels are a bit small, but the graphs are a useful summary of the distances in this abstract 12 dimensional space.

# In[15]:


fig = plt.figure(figsize=(15, 22.5))

# you can use this utility function to
# help you get the xticklabels in order
def list_in_order(alist, order):
    """Given a list 'alist' and a list of indices 'order'
    returns the list in the order given by the indices"""
    return [alist[i] for i in order]


def rank_plot(distances):
    # find the ordering of the distances
    order = np.argsort(distances)
    # bar plot them
    ax.bar(np.arange(len(distances)), distances[order])
    ax.set_xlabel("Distillery", fontsize=12)
    ax.set_ylabel("Distance to Glenlivet (in flavour space)", fontsize=12)
    ax.set_xticks(np.arange(86))
    ax.set_frame_on(False)
    # make sure the same order is used for the labels!
    ax.set_xticklabels(
        list_in_order(distilleries, order), rotation="vertical", fontsize=8
    )


# make the plots
ax = fig.add_subplot(3, 1, 1)
ax.set_title("$L_2$ norm", fontsize=16)
rank_plot(glenlivet_2)
ax = fig.add_subplot(3, 1, 2)
ax.set_title("$L_1$ norm", fontsize=16)
rank_plot(glenlivet_1)
ax = fig.add_subplot(3, 1, 3)
ax.set_title("$L_\infty$ norm", fontsize=16)
rank_plot(glenlivet_inf)

# removes ugly overlapping
plt.tight_layout()


# **Task A.1** 
# 
# Compute the $L_2$ distance between Glenfiddich and Lagavulin's flavour profiles. 
# - Store in a variable called `glenfiddich_lagavulin`

# In[16]:


### BEGIN SOLUTION
glenfiddich_lagavulin = np.linalg.norm(whisky[distilleries.searchsorted('Lagavulin')] - whisky[distilleries.searchsorted('Glenfiddich')], 2)
### END SOLUTION


# In[17]:



with tick.marks(2):        
    assert(check_hash(glenfiddich_lagavulin, ((), 34.70608276182724)))


# **Task A.2** 
# Which distillery is closest to Ardbeg's flavour profile in the $L_\infty$ norm? Store the distillery **name** in the variable in `like_ardbeg`. Note: compute this - do not hardcode it.
# 

# In[18]:


### BEGIN SOLUTION
like_ardbeg = distilleries[
    np.argsort(np.linalg.norm(whisky[distilleries.searchsorted('Ardbeg')] - whisky, ord=np.inf, axis=1))[1]]
print(like_ardbeg)
### END SOLUTION


# In[19]:


print("The distillery most like Ardbeg (according to the L_inf norm) is {distillery}.".
          format(distillery=like_ardbeg ))


# In[20]:



with tick.marks(2):        
    assert(case_crc(like_ardbeg)==1878156447)


# Which distillery is *geographically furthest* from Ardbeg (using L_2 norm)?

# In[21]:


### BEGIN SOLUTION
furthest_ardbeg = distilleries[
    np.argsort(np.linalg.norm(locations[distilleries.searchsorted('Ardbeg')] - locations, ord=2, axis=1))[-1]]
print(furthest_ardbeg)
### END SOLUTION


# In[22]:


# Sanity check. Validate that furthest_ardbeg is a str. 
# This test needs to pass for the hidden test to run correctly.
with tick.marks(0):        
    assert(type(furthest_ardbeg)==str)


# In[23]:


# Hidden test checking furthest_ardbeg [3 marks]


# **Note** from now on, use the $L_2$ norm if you need to compute any norms.
# 
# 
# ### Vector arithmetic
# A client says to you:
#     
# >    I'd like something a bit more "mellow" than Tormore, in the same way that Glenmorangie is more "mellow" than Bowmore.
# 
# Which whisky should you recommend? 
# 
# We can work this out:
# * What does more "mellow" mean? We don't have a "mellow" column.
#     * But we do have a reference point: Bowmore -> Glenmorangie is somehow "mellow"
#     * This "direction" between these flavour vectors is *also* a vector
# * How do we combine Tormore's flavour profile with "mellow"? We can compose vectors by addition.
# * How do we find a distillery that represents this profile? We can compute lengths of vectors using a norm.
# 
# **Task A.3** Compute:
# * `mellow` A vector representing what "mellow" is.
# * `hypothetical_flavour` A vector representing a hypothetical flavour that would be a more mellow version of Tormore.
# * `recommendation` the name of a specific distillery that we might recommend, as a string.
# 
# **Note: this question is not subjective, nor does it require any trial-and-error or knowledge about whisky. Answer it directly using vector arithmetic.**

# In[24]:


### BEGIN SOLUTION
tormore = whisky[distilleries.searchsorted('Tormore')]
bowmore = whisky[distilleries.searchsorted('Bowmore')]
glenmorangie = whisky[distilleries.searchsorted('Glenmorangie')]
mellow = glenmorangie - bowmore
hypothetical_flavour = tormore + mellow
distance_from_hypothetical = np.linalg.norm(whisky - hypothetical_flavour, 2, axis=1)
recommendation = distilleries[np.argmin(distance_from_hypothetical)]
### END SOLUTION


# In[25]:


print("I would recommend {distillery} as a more 'mellow' version of Tormore.".format(distillery=recommendation))


# In[26]:



with tick.marks(3):
    assert(check_hash(mellow, ((12,), -3.374787460507174)))


# In[27]:


with tick.marks(4):
    assert(check_hash(hypothetical_flavour,((12,), 49.63291973521592)))


# In[28]:


# Sanity check. Validate that furthest_ardbeg is a str. 
# This test needs to pass for the hidden test to run correctly.
with tick.marks(0):        
    assert(type(recommendation)==str)


# In[29]:


# Hidden test checking recommendation [4 marks]


# **Task A.4** A client wishes to taste whiskies than span a spectrum of flavours. You have been told:
# 
# * `Lagavulin` represents one end of this spectrum
# * `Auchentoshan` represents the other end of this spectrum.
# 
# Find a sequence of *five* distilleries, as evenly spaced across this spectrum as possible. Store the names in a list `tour`, which should begin "Lagavulin" and end in "Auchentoshan".

# In[30]:


### BEGIN SOLUTION
lagavulin = whisky[distilleries.searchsorted('Lagavulin')]
auchentoshan = whisky[distilleries.searchsorted('Auchentoshan')]

tour = []
for k in np.linspace(0, 1, 5):
    interpolated = k * auchentoshan + (1-k) * lagavulin
    tour.append(distilleries[np.argmin(np.linalg.norm(interpolated - whisky, axis=1, ord=2))])
    
print(tour) 

### END SOLUTION


# In[31]:


print("The recommended flavour tour from Lagavulin to Auchentoshan is:")

for distillery in tour:
    print("\t", distillery)


# In[32]:



with tick.marks(3):
    assert(case_crc(tour[0])==3089990555)
    assert(case_crc(tour[1])==1878156447)
    assert(case_crc(tour[2])==2088351511)    


# In[33]:


# A hidden test that tests the last two items in the list (3 marks)
# Note: You need to ensure that tour[3] and tour[4] contain the correct answers.


# ## Region flavours
# 
# We could say that the "representative" element of a collection of vectors was the one closest to the geometric centroid. This is given by the **mean vector** of a data set (computed below).
# 
# **Task A.5** Compute the mean vector of all of the flavour vectors. Use it to find the names of two distilleries:
# * `most_representative` The distillery with the **most** representative flavour profile
# * `least_representative` The distillery with the **least** representative flavour profile

# In[34]:


mean_vector = np.mean(whisky, axis=0)


# In[35]:


### BEGIN SOLUTION
most_representative = distilleries[np.argmin(np.linalg.norm(whisky - mean_vector, 2, axis=1))]
least_representative = distilleries[np.argmax(np.linalg.norm(whisky - mean_vector, 2, axis=1))]
### END SOLUTION


# In[36]:


print("The most representative whisky distillery is {most}, and the most unusual is {least}.".format(most=most_representative,
                                                                                           least=least_representative))


# In[37]:



with tick.marks(3):
    assert(case_crc(most_representative)==125187962)    


# In[38]:


# Hidden test checking least_representative [3 marks]
# Note: Make sure type(least_representative) is str


# 
# ### A map
# The code below will show a map of Scotland, with the distilleries in their correct positions. Different geographic regions have different characteristic flavour profiles.
# 
# One very distinctive region is the **island** region. This is a region bounded roughly by the box
# 
#     95000, 625000 -> 183000, 860000
#     
# in the same OS grid units used in the `locations` array. This is highlighted on the map below.
# 

# In[39]:


from whisky_map import draw_map, map_box

# draw each distillery label at the locations given.
ax = draw_map(locations, distilleries)

# show the island region
# draw a box in OS grid units
map_box(ax, 95000, 625000, 183000, 860000)


# **Task A.6**
# * Find all distilleries in island and use this to answer these questions:
# * `island_flavour`: Compute the most typical whisky flavour profile for island whiskies. (i.e. a 12 element vector).
# * `most_typical_island`: The specific name of the distillery from the islands that is most typical of that region;
# * `most_atypical_island`: The specific name of the distillery from the islands that is most atypical of that region (i.e. furthest from the typical);
# * `most_typical_non_island`: The specific name of the distillery from **outside** island that is most typical of that region;
# * `most_like_island`: The specific name of the distillery from **outside** the islands that is most like a typical island distillery.
# 
# **Note:** do not do any of this by hand. Write code.
# Hint: Boolean arrays.

# In[40]:


### BEGIN SOLUTION
is_island = (locations[:, 0] >= 95000) & (locations[:, 0] <= 183000) & (locations[:, 1] >= 625000) & (locations[:, 1] <= 860000)
island_distilleries = distilleries[is_island]

print(island_distilleries)
island_whisky = whisky[is_island]
island_flavour = np.mean(island_whisky, axis=0)
most_typical_island = island_distilleries[np.argmin(np.linalg.norm(island_whisky - island_flavour, 2, axis=1))]  # closest to mean
most_atypical_island = island_distilleries[np.argmax(np.linalg.norm(island_whisky - island_flavour, 2, axis=1))]  # closest to mean

non_island_distilleries = distilleries[~is_island]  # ~ stands for NOT
non_island_whisky = whisky[~is_island]
non_island_flavour = np.mean(non_island_whisky, axis=0)
most_typical_non_island = non_island_distilleries[np.argmin(np.linalg.norm(non_island_whisky - non_island_flavour, 2, axis=1))]  # closest to mean
most_like_island = non_island_distilleries[np.argmin(np.linalg.norm(non_island_whisky - island_flavour, 2, axis=1))] # non-island whisky closest to island mean
### END SOLUTION


# In[41]:



with tick.marks(2):
    assert(check_hash(island_flavour, ((12,), 104.99538270254608)))


# In[42]:


print("The most typical whisky of the island region is {most}.".format(most=most_typical_island))
print("The most atypical whisky of the island region is {most}.".format(most=most_atypical_island))
print("The most typical non-island whisky is {most}.".format(most=most_typical_non_island))
print("The non-island whisky most like island whiskies is {nonisland}.".format(nonisland=most_like_island))       


# In[43]:


with tick.marks(2):
    assert(case_crc(most_typical_island)==3459837550)


# In[44]:


with tick.marks(2):
    assert(case_crc(most_atypical_island)==999830981)


# In[45]:


# Hidden test checking most_typical_non_island [4 marks]
# Note: Make sure most_typical_non_island is a str


# In[46]:


# Hidden test checking most_typical_non_island [5 marks]
# Note: Make sure most_like_island is a str


# ---

# 
# # Task B: Eigendecompositions and whitening
# This part uses ideas from Week 3 of the course. You may wish to wait until after the Week 3 lecture to attempt this section. 
# 
# You are welcome to attempt it in advance, but you will have to do your own research.
# 
# 

# ## Focusing data
# <img src="imgs/drop.jpg" width="40%"> <br><br>*~[Image](https://flickr.com/photos/predi/236902022 "just a droplet, but upside down") by [Predi](https://flickr.com/people/predi) shared [CC BY-ND](https://creativecommons.org/licenses/by-nd/2.0/)*
# 
# This part will use  matrix decompositions to form abstract "lenses" that let us see data from different perspectives. This will let us pull out hidden structure and translate among representations.

# ## Demeaning
# The dataset that we have is unnormalised. It is a set of ratings, 0-4, and many of the attributes rated are very correlated (e.g. `smoky` and `medicinal`). Many of the ratings are also on quite different scales, with `tobacco` being much less likely to be rated 4 than `floral`.
# 
# It is easier to work with normalised data. 
# 
# **Task B.1**
# Compute:
# * `mean_vector` the average flavour profile (you should have this from part A already)
# * `demeaned_whisky` that has the mean flavour vector removed. 
# 
# 

# In[47]:


### BEGIN SOLUTION
mean_vector = np.mean(whisky, axis=0)
demeaned_whisky = whisky - mean_vector
### END SOLUTION


# In[48]:


## Show the mean vector as an image strip
## Remember: this represents a point in space
fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(1,1,1)
img = ax.imshow(mean_vector[None,:], cmap='viridis', vmin=0, vmax=4)
ax.set_xticklabels(columns)
ax.set_xticks(np.arange(len(columns)))
ax.set_yticks([])
ax.set_title("Mean vector of flavour profile")
fig.colorbar(img, orientation='horizontal')

with tick.marks(4):
    assert(check_hash(demeaned_whisky, ((86, 12), -12996.960309576743)))


# **Task B.2** Compute the **covariance matrix** of the *demeaned* data. Call this `whisky_cov`. 
# 
# The code below will show you this matrix as an image. **Note**: this should be a 12x12 matrix!
# 
# The covariance matrix tells us how different columns of the dataset are correlated (co-vary) with each other.

# In[49]:


### BEGIN SOLUTION
whisky_cov = np.cov(demeaned_whisky, rowvar=False) # rowvar=False means that the variables are columns, not rows
# np.cov(demeaned_whisky.T) # alternatively, we can transpose the matrix so that the variables are rows
# The values on the diagonal show the variance (std^2) of each variable. This hasn't been normalised yet.
### END SOLUTION


# In[50]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
img = ax.imshow(whisky_cov, cmap='seismic', vmin=-1.5, vmax=1.5)
ax.set_xticks(np.arange(12))
ax.set_yticks(np.arange(12))
ax.set_yticklabels(columns)
ax.set_xticklabels(columns, rotation='vertical')
ax.set_title("Covariance matrix")
fig.colorbar(img)

with tick.marks(4):
    assert(check_hash(whisky_cov, ((12, 12), 617.9256928561159)))


# From the image of the covariance matrix we can see several interesting things:
# * having `body` is negatively correlated with the `wine` flavour
# * being `smoky` is negatively correlated with having `sweetness`
# * `nutty` is positively correlated with `body`
# 
# Although the raters have provided 12 different categories, it seems these are not fully independent of each other.

# ## Eigendecomposition of the covariance matrix
# 
# We would like to create some new flavour categories that are independent of each other (so that we don't have categories like smoky and medicinal that are given similar numbers by the tasters). We can do this by combining the existing flavour categories.
# 
# By looking at the covariance matrix, we can get an idea of which flavours are most correlated or most independent. Let's analyse this in more detail. Recall that the covariance matrix can be represented by an ellipse whose primary axes are the **eigenvectors** of the covariance matrix. The eigenvectors are a set of *independent* directions in which the dataset varies. 
# 
# These are the **principal components** of the dataset. We can compute these from the covariance matrix by taking the eigendecomposition. Each eigenvector of the covariance matrix is a **principal component** and its importance is given by the square root of the absolute value of its corresponding eigenvalue.
# 
# Note that a 12 x 12 matrix will be represented by a 12-dimensional ellipsoid, which we can't really visualise, but we can visualise the relative importance of the eigenvectors. 
# 
# **Task B.3**
# * Compute all 12 of the eigenvectors of the covariance matrix, in order, with the *largest* corresponding eigenvalue first. Store these as a matrix of column vectors in `whisky_pc`. These eigenvectors are the principal components of the whisky data set.
# * Compute the square root of every eigenvalue (we can think of these as lengths of the prinicipal components) and store them in `whisky_pc_len`. Make sure they are in descending order from largest to smallest.

# In[51]:


### BEGIN SOLUTION
evals, evecs = np.linalg.eig(whisky_cov)

print(evals)  # 1D array of eigenvalues, unsorted
print(evals.argsort()[::-1])
evals_sorted = evals[
    evals.argsort()[::-1]
]  # sorted 1D array of eigenvalues (largest first)
print(evals_sorted)

plt.figure(1)
plt.title("Eigenvectors (columns), unsorted")
plt.imshow(evecs)  # matrix of eigenvectors, unsorted (eigenvectors are columns)
plt.colorbar()

plt.figure(2)
plt.title(
    "Eigenvectors (columns), sorted \n in order of decreasing eigenvalue \n (prinicpal component on left-hand side)"
)
plt.imshow(
    evecs.T[evals.argsort()[::-1]].T
)  # to sort the columns, take transpose, pick out rows in correct order, then transpose again
plt.colorbar()

whisky_pc = evecs.T[
    evals.argsort()[::-1]
].T  # sorted matrix of eigenvectors (largest eigenvalue first)
whisky_pc_len = np.sqrt(evals_sorted)
print(array_hash(whisky_pc_len))
### END SOLUTION


# In[52]:


## We can show the principal components as an image
## Each row is a principal component and shows a vector
## which represents the direction of variation. The first
## vector represents the largest component.
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(1,1,1)
# space out the principal components and scale them by the length of the components
img = ax.imshow(np.concatenate([(whisky_pc * whisky_pc_len), 
                                np.zeros_like(whisky_pc)], axis=0).T.reshape(24, -1).T, 
                cmap='seismic',         
                vmin=-2, vmax=2)

ax.set_yticks(np.arange(12))
ax.set_xticks(np.arange(0,24,2))
ax.set_xticklabels(["PC{i}".format(i=i) for i in np.arange(12)])
ax.set_yticklabels(columns)
ax.set_title("Principal components")
ax.set_frame_on(False)
fig.colorbar(img);


# In[53]:


## Show the principal component lengths of this dataset
## This shows how much of the variation in the dataset
## is "explained" by the variation along the corresponding direction.
## In this case, we can see that the first and second components are the largest.
## PC1 is mainly a combination of "smoky", "medicinal" and "body",
## whereas PC2 is mainly a combination of "honey", "nutty", "malty" and "body".
## Further down the list, PC4 is strongly "floral"
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.bar(np.arange(len(whisky_pc_len)), whisky_pc_len)
ax.set_xticks(np.arange(12))
ax.set_xticklabels(["PC{i}".format(i=i) for i in np.arange(12)])
ax.set_ylabel("$\sqrt{\lambda}$")
ax.set_frame_on(False)
ax.set_title("Component importances")


# In[54]:


with tick.marks(10):
    assert(check_hash(whisky_pc, ((12, 12), -116.3856770814677)))
    assert(check_hash(whisky_pc_len, ((12,), 49.2805781587291)))


# ## Projecting onto the principal components
# 
# This is interesting, but quite hard to interpret. One very useful technique is to project data onto a small number of principal components, to visualise the data. This forms a simplified version of the data, where the use of principal components means we can map the directions in the data which are most important to our visual axes. For example, we might map the first two principal components to a 2D $x,y$ plot.
# 
# This is a key technique in exploratory data analysis: **principal component analysis**. All it involves is using the principal components to find a simplified mapping onto a lower-dimensional space.
# 
# We can compute a projection of a dataset onto an arbitrary set of vectors by forming a matrix product:
# $$P = XV,$$
# 
# Where $P$ is an $N\times k$ matrix that is the result of the projection, $X$ is the $N \times D$ original data set and $V$ is an $D \times k$ matrix, each of whose *columns* is a vector that we want to project onto.
# 
# **Task B.4**
# Use this information to project the distillery data onto the first two principal components, and store the result in `whisky_projected_2d`. If you do this correctly, the plot below should show a 2D mapping of whisky flavours, where more distant distilleries in the map represent more distinct flavour styles.
# 
# 
# 

# In[55]:


### BEGIN SOLUTION
whisky_projected_2d = whisky @ whisky_pc[:, 0:2]
### END SOLUTION


# In[56]:


## Show the whisky distilleries laid out
## on the two first principal components,
## colouring the points according to the level of smokiness 
## (just to see that similar whiskies are indeed clustered together)
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1,1,1)
#ax.set_frame_on(False)
ax.set_xlabel("Principal component 1", fontsize=14)
ax.set_ylabel("Principal component 2", fontsize=14)
ax.set_title("Principal component analysis of whisky distillery flavour profiles", fontsize=16)
ax.scatter(whisky_projected_2d[:,0], whisky_projected_2d[:,1], c=whisky[:,columns['smoky']], s=40)
#ax.scatter(whisky_projected_2d[:,0], whisky_projected_2d[:,1], c=whisky[:,columns['nutty']], s=40)
for i,name in enumerate(distilleries):
    ax.text(whisky_projected_2d[i,0], whisky_projected_2d[i,1], name, fontdict={'size':12})


# In[57]:


with tick.marks(3):
    assert(check_hash(whisky_projected_2d, ((86, 2), 21159.91633246404)))


# **Now, repeat this exact process of Principal Component Analysis, but for the *geographic locations* instead of the flavour profiles, and compute `location_projected_2d`.** Hint: `location_projected_2d` should have shape (86,2).

# In[59]:


### BEGIN SOLUTION
mean_location = np.mean(locations,axis=0)

# Solution option 1: Undemean, Undemean
location_cov = np.cov(locations, rowvar=False)
evals, evecs = np.linalg.eig(location_cov)
location_pc = evecs.T[
    evals.argsort()[::-1]
].T 

location_projected_2_a = locations @ location_pc [:, 0:2]

# Solution option 2: Demean, Undemean
location_cov = np.cov(locations-mean_location, rowvar=False) # should not matter!
evals, evecs = np.linalg.eig(location_cov)
location_pc = evecs.T[
    evals.argsort()[::-1]
].T 

location_projected_2_b = locations @ location_pc [:, 0:2]

# Solution option 3: Undemean, demean
location_cov = np.cov(locations, rowvar=False)
evals, evecs = np.linalg.eig(location_cov)
location_pc = evecs.T[
    evals.argsort()[::-1]
].T 

location_projected_2_c = (locations-mean_location) @ location_pc [:, 0:2]


# Solution option 4: Demean, demean
location_cov = np.cov(locations-mean_location, rowvar=False)
evals, evecs = np.linalg.eig(location_cov)
location_pc = evecs.T[
    evals.argsort()[::-1]
].T 

location_projected_2_d = (locations-mean_location) @ location_pc [:, 0:2]


location_projected_2d = location_projected_2_a # for example

### END SOLUTION


# In[60]:


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1,1,1)
#ax.set_frame_on(False)
ax.set_xlabel("Geographical principal component 1", fontsize=14)
ax.set_ylabel("Geographical principal component 2", fontsize=14)
ax.set_title("Principal component analysis of whisky locations", fontsize=16)
ax.scatter(location_projected_2d[:,0], location_projected_2d[:,1], c=whisky[:,columns['smoky']], s=180)

for i,name in enumerate(distilleries):
    ax.text(location_projected_2d[i,0], location_projected_2d[i,1], name, fontdict={'size':12})


# In[61]:


# Sanity check of the size of location_projected_2d; it does not check the values of location_projected_2d. 
# If this fails, so will the hidden test!
with tick.marks(0):
    assert(check_hash(0.0*location_projected_2d, ((86, 2), 0.0)))    


# In[62]:


# Hidden test which checks location_projected_2d [9 marks]


# 
# We can normalise this data further. Whiten the dataset so that it has zero mean and unit covariance. This transforms our dataset so that it is centered on the origin (demeaning) and "spherical" (whitenening with covariance matrix). This is particularly useful if we are going to try and map from data in one vector space to another; having the data in standard scaling, with no offset and no correlation among dimensions makes the data easier to work with.
# 
# **Task B.5**
# Use the SVD to compute the inverse square root of the covariance matrix `whisky_cov`. Multiply the demeaned whisky matrix by this to produce `whitened_whisky`. This represents the data with the mean removed and all correlations eliminated.
# 
# 

# In[63]:


### BEGIN SOLUTION
u, s, v = np.linalg.svd(whisky_cov)
inv_sqrt = u @ np.diag(1.0 / np.sqrt(s)) @ v
whitened_whisky = demeaned_whisky @ inv_sqrt 
### END SOLUTION


# In[64]:


# This plot of the covariance matrix should now be perfectly diagonal
plt.imshow(np.cov(whitened_whisky.T), vmin=0, vmax=1, cmap='viridis')
plt.colorbar()

with tick.marks(2):
    assert(check_hash(whitened_whisky, ((86, 12), -13840.852324929752)))


# In[65]:


# show a plot of the whisky data
fig = plt.figure(figsize=(10,25))
ax = fig.add_subplot(1,1,1)
ax.set_title("Whitened, demeaned vectors")
# image plot
img = ax.imshow(whitened_whisky)
ax.set_yticks(np.arange(len(distilleries)))
ax.set_yticklabels(distilleries, rotation="horizontal", fontsize=12)

# put the x axis at the top
ax.xaxis.tick_top()
ax.set_xticks(np.arange(len(columns)))
ax.set_xticklabels(columns, rotation="vertical", fontsize=12)

# some horrific colorbar hackery to put in the right place
# don't worry about this bit!
cbaxes = fig.add_axes([0.37, 0.93, 0.28, 0.01])  
fig.colorbar(img, orientation='horizontal',  cax=cbaxes, ticks=np.arange(5))
cbaxes.xaxis.tick_top()


# ### Visualising in normalised space
# Now that we have the data normalised, we could also define a more sensible way to compare vectors in a high-dimensional space. The $L_2$ norm has significant problems in very high-dimensional spaces. A more sensible way to compare high-dimensional vectors is to look at the *angle* between them.
# 
# **Task B.6**
# Define a function `cosine(a, b)` that computes the cosine of the angle between two vectors `a` and `b`. 
# 
# N.B. This should be a value between -1 and 1.

# In[66]:


def cosine(a, b):
    ### BEGIN SOLUTION
    return np.dot(a, b) / (np.linalg.norm(a, 2) * np.linalg.norm(b, 2))
    ### END SOLUTION


# In[67]:



angles = np.array([cosine(whitened_whisky[i,:], whitened_whisky[j,:]) for i in range(len(whisky)) for j in range(len(whisky))])

with tick.marks(2):
    assert(check_hash(angles, ((7396,), 25457.146694199713)))


# ## A 3D visualisation
# We can use the cosine distances to show a different layout of whiskies; this time in terms of relative angles to two reference distillieries, in the whitened space. This has the advantage that all distances are normalised to the range [-1,1], and we have a good spread of points in the space. The plot below shows the angle with respect to 3 distilleries, as a 3D plot. You can compare with the plot using the unwhitened data to see the effect that normalisation has had, if you wish.
# 
# This is also an example of why visualising data in 3D is usually a bad idea.

# In[68]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1, projection='3d')

## Modify these to plot the flavours on different axes
ref_a = 'Glengoyne'
ref_b = 'Laphroig'
ref_c = 'Glenlivet'

## get the respective flavour vectors
reference_a = whitened_whisky[distilleries.searchsorted(ref_a)]
reference_b = whitened_whisky[distilleries.searchsorted(ref_b)]
reference_c = whitened_whisky[distilleries.searchsorted(ref_c)]

for name, flavour in zip(distilleries, whitened_whisky):
    ## compute angles to the references
    angle_a = cosine(reference_a, flavour)
    angle_b = cosine(reference_b, flavour)
    angle_c = cosine(reference_c, flavour)

    ax.scatter(angle_a, angle_b, angle_c, color='c', s=5)
    ax.text(angle_a, angle_b, angle_c, name, fontdict={"size":4}, alpha=0.5)
    
## fix up the plot
ax.set_xlabel("Angle with respect to {ref_a}".format(ref_a=ref_a))
ax.set_ylabel("Angle with respect to {ref_b}".format(ref_b=ref_b))
ax.set_zlabel("Angle with respect to {ref_c}".format(ref_c=ref_c))
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(-1.1, 1.1)
ax.set_title("Normalised flavour angles between whisky distilleries")


# ---

# ## C: A new competitor...? 
# 
# Consider the following problem: A newly founded distillery aims to produce a whisky similar to "Glenlivet" but would like 2.5 points more "body", 0.1 points more "sweetness" and 0.2 points fewer on the "tobacco" scale - otherwise, they are happy with the Glenlivet flavor profile.
# 
# Is it possible for the distillery to produce this whisky given their current production system has limited production capabilities (modeled as a *linear system*)?
# 

# **Task C.1**
# - Extract the flavour profile for Glenlivet from `demeaned_whisky` and store it in `demeaned_glenlivet`
# - Compute the new target flavour profile (based on the `demeaned_demeaned_glenlivet`) the new distillery is aiming for; store it in `demeaned_desired_flavour`
# 

# In[69]:


### BEGIN SOLUTION
demeaned_glenlivet = demeaned_whisky[distilleries=='Glenlivet',:][0]

demeaned_desired_flavour=demeaned_glenlivet.copy()
demeaned_desired_flavour[columns["body"]] = demeaned_desired_flavour[columns["body"]] + 2.5
demeaned_desired_flavour[columns["sweetness"]]  = demeaned_desired_flavour[columns["sweetness"]] + 0.1
demeaned_desired_flavour[columns["tobacco"]]  = demeaned_desired_flavour[columns["tobacco"]] -0.2

### END SOLUTION


# In[70]:


## Show the demeaned_glenlivet vector as an image strip
## Remember: this represents a point in space
fig = plt.figure(figsize=(11, 4))
ax = fig.add_subplot(1,1,1)
img = ax.imshow(demeaned_glenlivet[None,:], cmap='viridis', vmin=0, vmax=4)
ax.set_xticklabels(columns)
ax.set_xticks(np.arange(len(columns)))
ax.set_yticks([])
ax.set_title("Glenlivet")
fig.colorbar(img, orientation='horizontal')

## Show the demeaned_glenlivit vector as an image strip
fig = plt.figure(figsize=(11, 4))
ax = fig.add_subplot(1,1,1)
img = ax.imshow(demeaned_desired_flavour[None,:], cmap='viridis', vmin=0, vmax=4)
ax.set_xticklabels(columns)
ax.set_xticks(np.arange(len(columns)))
ax.set_yticks([])
ax.set_title("Desired flavour profile")
fig.colorbar(img, orientation='horizontal')


# In[71]:



with tick.marks(2):
    assert(check_hash(demeaned_glenlivet, ((12,), 35.3928040241586)))


# In[72]:



with tick.marks(2):
    assert(check_hash(demeaned_desired_flavour, ((12,), 38.66863915527948)))


# **Task C.2**
# 
# The new distillery has a new but simple production facility and only limited access to quality ingredients (specific malt types), which limits their flexibility. They only have five specific parameters they can easily control, such as the age of the barrels in which they age the whiskey and certain pressure and temperature settings during the distillation and aging. In total, they have five such (continuous) parameters, which we can **conveniently** collect in a vector $\mathbf{x}\in\mathbb{R}^5$; each parameter has a unique effect on the resulting flavor.
# 
# We also happen to know a set of equations govering the input-output relationship of the current Whisky distillery. When we  produce our Whisky with a certain $\mathbf{x}$ as input, we know the result of the distillation in terms of the flavors. The input-output relations are given by the follwing equations:
# 
# $$\begin{gathered}
#   {y_1} = 1.01 {x_1}{\text{  + }}{x_2}{\text{  +  }}{x_5}{\text{ }} \hfill \\
#   -2{x_1}+{y_2} = 0.02{x_2}{\text{  + }}{x_3}{\text{ }} - {x_4}{\text{ +  }}{x_5}{\text{ }} \hfill \\
#   {y_3} =  - 2{x_1}{\text{  + }}{x_2}{\text{ +  }}1.03{x_3}{\text{  + }}{x_4}{\text{ }} - {x_5} \hfill \\
#   {y_4} = {x_1}{\text{ }} + {x_2}{\text{  + }}{x_3}{\text{ }} - 2.04{x_4}{\text{  }} \hfill \\
#   {y_5} =  - 0.005{x_5}{\text{ }} \hfill \\
#   {y_6} =  - {x_1}{\text{  +  }}{x_2}{\text{ }} - 2{x_3}{\text{ +  }}{x_5} \hfill \\
#   -{x_2}{\text{ -  }}{x_3}{\text{ }}+ {y_7} = 0 \hfill \\
#   {y_8} = {x_2}{\text{ }} - {x_3}{\text{  +  }}2.05{x_5} \hfill \\
#   {y_9} = 4{x_1}{\text{  + }}{x_2} - {x_3}{\text{ }} - 1.04{x_4}{\text{ }} - {x_5} \hfill \\
#   {y_{10}} = 3{x_1}{\text{ }} - {x_2}{\text{ }} - 0.03{x_3} \hfill \\
#   {y_{11}} =  - {x_1}{\text{  + }}1.02{x_2}{\text{ }} - 3{x_3} \hfill \\
#   {y_{12}} = 3.01{x_1}{\text{  + }}2{x_2}{\text{  +  }}{x_3}{\text{  +  }}{x_4}{\text{  + }}2.05{x_5} \hfill \\ 
# \end{gathered} $$
# 
# Where $\mathbf{x}=[x_1,x_2,...,x_{12}]$ are the parameters and $\mathbf{y}=[y_1,y_2,...,y_{12}]$ is the desired flavor profile. For example, if we know the value of the system parameters $x_1$ and $x_2$ and $x_5$ we know what the "body" of the Whisky will be once it has finished aging and has been bottled.
# 
# In short, when we apply the set of equations to our process parameters vector, $\mathbf{x}$, we know the resulting flavour. This is a *system of linear equations* (or just a *linear system*) which can be written $A\mathbf{x} = \mathbf{y}$ where $\mathbf{y}$ is the vector stored in `demeaned_desired_flavour` and $\mathbf{x}$ is the vector of parameters we need to produce a whisky with a certain flavour profile.
# 

# 
# **Task B.3**
# 
# From the set of coupled equations above, identify $A$ and create a $12x5$ matrix, and store it in a numpy array called `A_whisky_process` . This is a manual and tedious task but important to understand! It should have values such that we can apply it to $\mathbf{x}$ and get $\mathbf{y}$, i.e., $A\mathbf{x} = \mathbf{y}$ where $\mathbf{y}$ is the vector `demeaned_desired_flavour`. You should not be applying any manual solving techniques to $A$ and $\mathbf{y}$ but simply inspect the individual equations and identify the correct values of the corresponding row in $A$.

# In[73]:


### BEGIN SOLUTON 
A_whisky_process = np.array([
    [ 1.01,     1,     0 ,       0 ,    1],  #1
    [ 2 ,      0.02,  1,       -1 ,    1],  #2
    [-2,       1,     1.03 ,    1 ,   -1],  #3
    [ 1,       1,     1,       -2.04 ,    0],  #4
    [ 0,       0,     0,        0 ,   -0.005],  #5
    [-1,       1,    -2,        0 ,    1],  #6    
    [ 0,       1,     1,        0 ,    0],  #7    
    [ 0,       1,    -1,        0 ,    2.05],  #8    
    [ 4,       1 ,   -1,       -1.04 ,   -1],  #9    
    [ 3,      -1,    -0.03 ,       0 ,    0],  #10    
    [-1,       1.02,    -3,        0 ,    0],  #11
    [ 3.01,    2 ,    1 ,       1 ,    2.05]]) #12

A_whisky_process
### END SOLUTION


# In[74]:


## Sanity check - size of A_whisky_process (0 marks)
with tick.marks(0): # a better test/hash
    assert(check_hash( 0*A_whisky_process , ((12, 5), 0)))


# In[75]:


## Visible, autograded assesment (advanced hash - WARNING: does not check the correct size of A_whisky_process)


with tick.marks(5): # ADVANCED HASH; DOES NOT CHECK FOR CORRET SIZE OF A_whisky_process
    assert(check_hash( A_whisky_process @ (np.linspace(1,2515.337,12*5)).reshape((5,12)), ((12, 12), 14089780.468302336)))


# **Task C.4**
# 
# We now look to find the set of ingredients and process parameters (i.e. $\mathbf{x})$ that will hopefully result in the new distillery being able to produce the new flavour.
# 
# $A$ is asymmetrical and this problem does not have an exact solution. We can however use the pseudo-inverse to give the closest result according to the L2 norm. In other words, it will find the vector $\mathbf{\hat x}$ that minimises the distance $\left\| {A\mathbf{\hat x}−\mathbf{y}} \right\|_2$:
# 
# - use `np.linalg.pinv` to compute an estimate the of process parameters, $\mathbf{\hat x}$, such that $A \mathbf{\hat x}\approx \mathbf{y}$ you'll need to use to produce the Whisky with the desired flavour (see the lecture notes on the pseudo inverse). Store the result in `xhat` with shape `(5,)`.
# - compute the actual flavour resulting from using the process parameters, $ \mathbf{\hat x}$, and store it in `demeaned_actual_flavour` (only for visualisation).
# - compute the L2 norm of the difference vector between the actual and desired flavour vector and store it in `error_l2` as a scalar.
# 

# In[76]:


### BEGIN SOLUTION 
A_pinv = np.linalg.pinv(A_whisky_process) 
xhat = A_pinv @ demeaned_desired_flavour
demeaned_actual_flavour = A_whisky_process @ xhat
error_l2 = np.linalg.norm(demeaned_actual_flavour - demeaned_desired_flavour)
print("l2 err: " ,error_l2)

### END SOLUTION 


# In[77]:


## Show the demeaned_glenlivet vector as an image strip
## Remember: this represents a point in space
fig = plt.figure(figsize=(11, 4))
ax = fig.add_subplot(1,1,1)
img = ax.imshow(demeaned_glenlivet[None,:], cmap='viridis', vmin=0, vmax=4)
ax.set_xticklabels(columns)
ax.set_xticks(np.arange(len(columns)))
ax.set_yticks([])
ax.set_title("Glenlivet")
fig.colorbar(img, orientation='horizontal')

## Show the demeaned_desired_flavour vector as an image strip
fig = plt.figure(figsize=(11, 4))
ax = fig.add_subplot(1,1,1)
img = ax.imshow(demeaned_desired_flavour[None,:], cmap='viridis', vmin=0, vmax=4)
ax.set_xticklabels(columns)
ax.set_xticks(np.arange(len(columns)))
ax.set_yticks([])
ax.set_title("Desired flavour profile")
fig.colorbar(img, orientation='horizontal')

## Show the demeaned_actual_flavour vector as an image strip
fig = plt.figure(figsize=(11, 4))
ax = fig.add_subplot(1,1,1)
img = ax.imshow(demeaned_actual_flavour[None,:], cmap='viridis', vmin=0, vmax=4)
ax.set_xticklabels(columns)
ax.set_xticks(np.arange(len(columns)))
ax.set_yticks([])
ax.set_title("Actual/possible flavour profile")
fig.colorbar(img, orientation='horizontal')


# In[78]:


## Sanity check - size of xhat (0 marks)
with tick.marks(0): 
    assert(check_hash( 0*xhat , ((5,), 0)))


# In[79]:


## Hidden, autograded assesment of xhat (4 marks)


# In[80]:


## Sanity check - size of error_l2 (0 marks)
with tick.marks(0): 
    assert(check_hash( 0*error_l2 , ((), 0)))


# In[81]:


## Hidden, autograded assesment of the error_l2 variable (2 marks)


# We can now (hopefully) answer the question whether the newly founded distillery is currently able to produce a Whisky with the desired flavor profile - or should they perhaps look to improve their production facilities (or choose a different business entirely)...?
# 

# ---

# # Submission on Moodle

# 
# We will generate the **one** pdf file you'll need to submit along with the notebook:
# 
# *Note*: you do not need to worry about the formatting etc (that's predetermined); just make sure all your explanations are readable in the pdf and your'll be fine!
# 

# In[82]:


## Report generation - YOU MUST YOU RUN THIS CELL !
#
# Ignore warnings regarding fonts
#

from matplotlib.backends.backend_pdf import PdfPages

# File 1: declaration of originality with system info
try:
    f = open('uofg_declaration_of_originality.txt','r')
    uofg_declaration_of_originality = f.read()
except: 
    uofg_declaration_of_originality = "uofg_declaration_of_originality not present in cwd"

try:
    student_id.lower()
except: 
    student_id="NORESPONSE"
try:
    student_typewritten_signature.lower()
except: 
    student_typewritten_signature="NORESPONSE"

fn = ("idss_lab_01_complinalg_%s_declaration.pdf" % (student_id.lower()))
fig_dec = plt.figure(figsize=(10, 12)) 
fig_dec.text(0.1,0.1,("%s\n\n Student Id %s\n\n Typewritten signature: %s\n\n UUID System: %s" % (uofg_declaration_of_originality,student_id, student_typewritten_signature, uuid_system)))
 
# Combined: 
fn = ("idss_lab_01_complinalg_%s_combined_v20202021a.pdf" % (student_id))
pp = PdfPages(fn)
pp.savefig(fig_dec)
pp.close()

with tick.marks(0):  # have you generated the combied file...? you don't actually get any credit for this just confirmation that the file has been generated
    assert(os.path.exists(fn))


# **You must (for full or partial marks) submit via Moodle:**
# 
# - this notebook (completed) after "Restart and rerun all":
#     - `idss_lab_01_complinalg_v20202021a.ipynb`
#     
# - the pdf (autogenerated):
#      - `idss_lab_01_complinalg_[YOUR STUDENT ID]_combined_v20202021a.pdf`)

# ---

# # Appendix: Marking Summary (and other metadata)
# #### - make sure that the notebook runs without errors (remove/resolve the `raise NotImplementedError()`) and "Restart and Rerun All" cells to get a correct indication of your marks.

# In[83]:


print("Marks total : ","100")
print("Marks visible (with feedback): ","70")
print("Marks hidden (without feedback): ","30")
print("Marks autograded (hidden+visible): ","100")
print("\nThe fraction below displays your performance on the autograded part of the lab that is visible with feedback (only valid after `Restart and Run all`:")
tick.summarise_marks() # 

print("- the autograded (and visible) marks account for at least 50% of the total lab assesment.")


# In[ ]:





# In[ ]:




