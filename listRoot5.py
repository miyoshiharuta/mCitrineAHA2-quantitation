'''
Created on Feb 19, 2017

@author: Surf32
'''
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib._cm import _CMRmap_data
import matplotlib
from builtins import range
from bokeh.models.ranges import Range
matplotlib.rcParams.update({'font.size': 5})
import pandas as pd
import skimage.morphology
from  czifile import CziFile
import osDB
import pyqtgraph

#create an excel file based on the images present
#get list of files containing following string
tartgetfiles = '.czi'
#targetdir = r'C:\Users\Surf32\Desktop\ResearchDSKTOP\Miyoshi\ByDay'
targetdir = r'C:\Users\Surf32\Desktop\ResearchDSKTOP\Miyoshi\5List\051517toDan'
czifiles, indx = osDB.getFileContString(targetdir, tartgetfiles)
# generate names for each row
names = []
pathfile =[]
for row in czifiles.index:
    names.append(czifiles.loc[row][:-4])
    pathfile.append(os.path.join(targetdir, czifiles.loc[row]))
#generate a pandas dataframe listing files and paths
xlFrame = pd.DataFrame(pathfile, index = names, columns =[ 'Pathfile'])
#save xlsfile
xlspath = os.path.join(targetdir, 'SummaryFile.xlsx')
xlFrame.to_excel(xlspath)

#load images
xlFrame['Image'] = ''
for row in xlFrame.index:
    #xlFrame['Image'].loc[row] = tifffile.imread( xlFrame['Pathfile'].loc[row])
    with CziFile(xlFrame['Pathfile'].loc[row]) as czi:
        img = np.squeeze(czi.asarray())
        img = np.rollaxis(img, 2,0)
        img = np.flipud(img)
        img = np.rollaxis(img, 0,3)
        xlFrame['Image'].loc[row] = img

#normal intensity values within image
def normalizeImage(img):    
    img = img - np.min(img)
    return img / np.max(img)

#determines the range
def setrange(x, xrange, limits):
    #x = number in index
    #range = the + and - of the given index
    #limits = [xin xmax] in possible range
    if x-xrange < limits[0]:
        x1 = limits[0]
    else:
        x1 = x-xrange
    if x + xrange > limits[1]:
        x2 = limits[1]
    else:
        x2 = x +xrange
    return np.arange(x1, x2)
    
        

    
#create a figure for each image
thresh = 75/255.0
minThresh=50 /255.0
width = 0.25
xrange = 5
outputfolder = r'C:\Users\Surf32\Desktop\ResearchDSKTOP\Miyoshi\5List\Individual'
for row in xlFrame.index:
    #plt.close('all')
    #start figure
    fig1 = plt.figure(figsize = [3.5, 5], dpi=300) 
    #draw image
    ax1 = fig1.add_axes([0.2, 0.6, width, width])
    img = normalizeImage(xlFrame['Image'].loc[row])
    limits = [0, img.shape[2]]
    #imgsh = ax1.imshow(np.rot90(np.max(img, axis=0)), aspect = 'equal')
    imgsh = ax1.imshow(np.max(img, axis=0), aspect = 'equal')
    imgsh.set_cmap('CMRmap')
    ax1.get_xaxis().set_tick_params(direction = 'out')
    #ax1.axis('off')
    ax1.yaxis.set_visible(False)
    cbar = fig1.colorbar(imgsh, location = 'top')
    cbar.ax.tick_params(labelsize = 6)
    
    #make plots
    abovethresh = img.copy()
    abovethresh[img < thresh] =0
    abovethresh[img >= thresh] = 1
    ax2 = fig1.add_axes([0.2, 0.35, width,width])
    #ax2.imshow(np.max(abovethresh, axis =0))
    #ax2.plot([400, 400], [0, 500], color ='k')
    
    #plot number of pixels above threshold
    abovethreshSum=[]
    for x in range(img.shape[2]):
        #ax2.plot(x, np.sum(abovethresh[:, :, x]))
        abovethreshSum.append(np.sum(abovethresh[:, :,setrange(x, xrange, limits)]))
    ax2.plot(abovethreshSum, color = 'r', label = str(thresh) + ' < pixels')
    
    #plot number of pixels below threshold
    belowthresh = img.copy()
    belowthresh[img < thresh] = 1
    belowthresh[img >= thresh] =0
    belowthresh[img < minThresh] =0
    belowthreshSum = []
    for x in range(img.shape[2]):
        belowthreshSum.append(np.sum(belowthresh[:, :, setrange(x, xrange, limits)]))
    ax2.plot(belowthreshSum, color = 'c', label = str(minThresh) + ' < pixels < ' + str(thresh), alpha = 0.5)
    
    #plot number of pixels above minThreshold
    allPixels = img.copy()
    allPixels[img >= minThresh] = 1
    allPixels[img < minThresh] =0
    allPixelsSum = []
    for x in range(img.shape[2]):
        allPixelsSum.append(np.sum(allPixels[:, :, setrange(x, xrange, limits)]))
    ax2.plot(allPixelsSum, color = 'k', label = str(minThresh) + ' < pixels ', alpha = 0.4)
    ax2.set_xlim([0 , 512]) 
    ax2.set_ylabel('Pixel Volume') 
    #ax2.legend(loc =3, prop={'size': 6})
    
    ax3 = fig1.add_axes([0.2, 0.05, width, width])
    stdev = []
    for x in range(img.shape[2]):
        stdev.append(np.std(img[:, :, setrange(x, xrange, limits)]))
    ax3.plot(stdev, color = 'k', label = str(minThresh) + ' < pixels ', alpha = 0.4)
    ax3.set_xlim([0 , 512]) 
    ax3.set_ylabel('Stand. Deviat.')
    plt.draw()
    plt.savefig(os.path.join(outputfolder,  row + '.jpeg'))
    plt.close(fig1)
 


#generate images giving percent pixels 
thresh = 75/255.0
minThresh=50 /255.0
width = 0.25
for row in xlFrame.index:
    #plt.close('all')
    #start figure
    fig1 = plt.figure(figsize = [3.5, 5], dpi=300) 
    #draw image
    ax1 = fig1.add_axes([0.2, 0.6, width, width])
    img = normalizeImage(xlFrame['Image'].loc[row])

    #imgsh = ax1.imshow(np.rot90(np.max(img, axis=0)), aspect = 'equal')
    imgsh = ax1.imshow(np.max(img, axis=0), aspect = 'equal')
    imgsh.set_cmap('CMRmap')
    ax1.get_xaxis().set_tick_params(direction = 'out')
    #ax1.axis('off')
    ax1.yaxis.set_visible(False)
    cbar = fig1.colorbar(imgsh, location = 'top')
    cbar.ax.tick_params(labelsize = 6)
    
    #make plots
    #get number of pixels above minThreshold
    allPixels = img.copy()
    allPixels[img >= minThresh] = 1
    allPixels[img < minThresh] =0
    allPixelsSum = []
    for x in range(img.shape[2]):
        allPixelsSum.append(np.sum(allPixels[:, :, setrange(x, xrange, limits)]))
        
        
    abovethresh = img.copy()
    abovethresh[img < thresh] =0
    abovethresh[img >= thresh] = 1
    ax2 = fig1.add_axes([0.2, 0.35, width,width])
    #ax2.imshow(np.max(abovethresh, axis =0))
    #ax2.plot([400, 400], [0, 500], color ='k')
    
    #plot number of pixels above threshold as percent
    abovethreshSum=[]
    for x in range(img.shape[2]):
        #ax2.plot(x, np.sum(abovethresh[:, :, x]))
        abovethreshSum.append(np.sum(abovethresh[:, :, setrange(x, xrange, limits)]))
    ax2.plot(np.array(abovethreshSum)/ np.array(allPixelsSum), color = 'r', label = '{0:.3}'.format(thresh) + ' < pixels')
    
    #plot number of pixels below threshold
    belowthresh = img.copy()
    belowthresh[img >= thresh] =0
    belowthresh[img < minThresh] =0
    belowthresh[belowthresh >= minThresh] = 1
    
    belowthreshSum = []
    for x in range(img.shape[2]):
        belowthreshSum.append(np.sum(belowthresh[:, :, setrange(x, xrange, limits)]))
        #elowthreshSum.append(np.sum(belowthresh[:, :, x]))
    ax2.plot(np.array(belowthreshSum)/ np.array(allPixelsSum), color = 'c', label = '{0:.3}'.format(minThresh) + ' < pixels < ' + '{0:.3}'.format(thresh), alpha = 0.5)
    
    #plot number of pixels below threshold
    ax2.plot((np.array(belowthreshSum)  - np.array(abovethreshSum))/np.array(allPixelsSum) , color = 'b', label = 'ratio (low- high) /all', alpha = 0.5)
    
    ax2.set_ylabel('Fraction total pixels')
    
    ax2.legend()
    plt.savefig(os.path.join(outputfolder,  row+ 'Percent.jpeg'))
    plt.close(fig1)
    
#combine roots from dim and bright conditions
conditions  = {'Dim': '3', 'Bright': '1'}
combined = pd.DataFrame( index = list(conditions.keys()), columns = ['row'])
for cindex in combined.index:
    combined['row'].loc[cindex] = []
    for imgrow in xlFrame.index:
        if conditions[cindex] in imgrow:
            combined['row'].loc[cindex].append(imgrow)
#get the maximum dimensions in y
xsize = []
for cdx in xlFrame.index:
    xsize.append(xlFrame['Image'].loc[cdx].shape[2])

#make numpy arrays to fill in data
newcolumns = ['AllPixels', 'Above', 'Below', 'Ratio']
for cc in newcolumns:
    combined[cc] = ''
    
for cindex in combined.index:
    for cc in newcolumns:
        combined[cc].loc[cindex] = np.zeros((len(combined['row'].loc[cindex]), np.max(xsize)))
     
for cindex in combined.index:
    for i, anim in enumerate(combined['row'].loc[cindex]):  
        #get pixels above threshold
        img = normalizeImage(xlFrame['Image'].loc[anim].copy())
        allPixels = normalizeImage(xlFrame['Image'].loc[anim].copy())
        allPixels[img >= minThresh] = 1
        allPixels[img < minThresh] =0
        allPixelsSum = []
        for x in range(img.shape[2]):
            allPixelsSum.append(np.sum(allPixels[:, :, setrange(x, xrange, limits)]))
        combined['AllPixels'].loc[cindex][i, 0:len(allPixelsSum)] = allPixelsSum #all the pixels above the minimum threshold
        
        #get the number of pixels above maximum threshold
        abovethresh = normalizeImage(xlFrame['Image'].loc[anim].copy())
        abovethresh[img < thresh] =0
        abovethresh[img >= thresh] = 1
        abovethreshSum=[]
        for x in range(img.shape[2]):
            #ax2.plot(x, np.sum(abovethresh[:, :, x]))
            abovethreshSum.append(np.sum(abovethresh[:, :, setrange(x, xrange, limits)]))
        combined['Above'].loc[cindex][i, 0:len(abovethreshSum)] = abovethreshSum

    
        #get the number of pixels above minimum threshold but below maximum threhs
        belowthresh = normalizeImage(xlFrame['Image'].loc[anim].copy())
        belowthresh[img < thresh] = 1
        belowthresh[img >= thresh] =0
        belowthresh[img < minThresh] =0
        belowthreshSum = []
        for x in range(img.shape[2]):
            belowthreshSum.append(np.sum(belowthresh[:, :, setrange(x, xrange, limits)]))
        combined['Below'].loc[cindex][i, 0:len(belowthreshSum)] = belowthreshSum

#graph classes in each class
def SEbarsToLinePlot(data, color, axes):
    #data = np array ploted along axis=1
    for i in range(data.shape[1]):
        se = np.nanstd(data[:, i]) / np.sqrt(data.shape[0])
        me = np.nanmean(data[:, i], axis=0)
        axes.plot([i, i], [me-se, me+se], alpha = 0.4, color = color, linewidth=0.2)
thresh = 0.35
minThresh=0.1
width = 0.25
xrange = 11

cindex = 'Dim'
width =0.3
fig1 = plt.figure(figsize = [4, 3.5], dpi=600) 
ax1 = fig1.add_axes([0.1, 0.2, width, width])
color = [0.9333, 0.9333, 0]
linewidth =1
ax1.plot(np.nanmean(combined['Above'].loc[cindex] / combined['AllPixels'].loc[cindex], axis=0), color = color, label = '{0:.3}'.format(thresh) + ' < pixels', linewidth =linewidth)
SEbarsToLinePlot(combined['Above'].loc[cindex] / combined['AllPixels'].loc[cindex], color=color, axes=ax1)
color = [1, 0, 0]
ax1.plot(np.nanmean(combined['Below'].loc[cindex] / combined['AllPixels'].loc[cindex], axis=0), color = color, label = '{0:.3}'.format(minThresh) + ' < pixels <' +  '{0:.3}'.format(thresh), linewidth = linewidth)
SEbarsToLinePlot(combined['Below'].loc[cindex] / combined['AllPixels'].loc[cindex], color=color, axes=ax1)

#plot number of pixels below threshold
color = [0.098039, 0.098039, 0.439216]
ax1.plot(np.nanmean((combined['Below'].loc[cindex]  - combined['Above'].loc[cindex])/combined['AllPixels'].loc[cindex], axis=0) , color = color, label = 'ratio (low- high) /all', linewidth = linewidth)
SEbarsToLinePlot((combined['Below'].loc[cindex]  - combined['Above'].loc[cindex])/combined['AllPixels'].loc[cindex], color=color, axes=ax1)
ax1.set_ylabel("Pixel Ratio Above Threshold")
micronsPerPixel = 212.5 / 512
label1 = ax1.get_xticks()
labeln = np.round(label1 * micronsPerPixel).astype(np.int)
ax1.set_xticklabels(labeln)
#ax1.legend()

cindex = 'Bright'
ax2 = fig1.add_axes([0.4, 0.2, width, width])

color = [0.9333, 0.9333, 0]
linewidth =1
ax2.plot(np.mean(combined['Above'].loc[cindex] / combined['AllPixels'].loc[cindex], axis=0), color = color, label = '{0:.3}'.format(thresh) + ' < pixels', linewidth =linewidth)
SEbarsToLinePlot(combined['Above'].loc[cindex] / combined['AllPixels'].loc[cindex], color=color, axes=ax2)
color = [1, 0, 0]
ax2.plot(np.mean(combined['Below'].loc[cindex] / combined['AllPixels'].loc[cindex], axis=0), color = color, label = '{0:.3}'.format(minThresh) + ' < pixels<' +  '{0:.3}'.format(thresh), linewidth = linewidth)
SEbarsToLinePlot(combined['Below'].loc[cindex] / combined['AllPixels'].loc[cindex], color=color, axes=ax2)

#plot number of pixels below threshold
color = [0.098039, 0.098039, 0.439216]
ax2.plot(np.nanmean((combined['Below'].loc[cindex]  - combined['Above'].loc[cindex])/combined['AllPixels'].loc[cindex], axis=0) , color = color, label = '(low- high) /all', linewidth = linewidth)
SEbarsToLinePlot((combined['Below'].loc[cindex]  - combined['Above'].loc[cindex])/combined['AllPixels'].loc[cindex], color=color, axes=ax2)
#ax1.legend()
ax2.set_xlabel('Distance from Tip (microns)')
ax2.xaxis.set_label_coords(0.02, -0.2)
ax2.set_yticklabels([])
label1 = ax2.get_xticks()
labeln = np.round(label1 * micronsPerPixel).astype(np.int)
ax2.set_xticklabels(labeln)

ax2.legend(loc=9, bbox_to_anchor=(-0, -.3), ncol=3)

ax3 = fig1.add_axes([0.1, 0.5, width, width])
#the dim images have a reduced y axis size compared to bright, so have to normalize axis
img = np.max(normalizeImage(xlFrame['Image'].loc[combined['row'].loc['Dim'][2]]), axis=0)
imgN = np.zeros(np.max(normalizeImage(xlFrame['Image'].loc[combined['row'].loc['Bright'][2]]), axis=0).shape, dtype = img.dtype)   
imgN[int((imgN.shape[0]-img.shape[0])/2) : -int((imgN.shape[0]-img.shape[0])/2), :] = img                     

imgD = ax3.imshow(imgN)
#ax3.axis('off')
imgD.set_cmap('CMRmap')
ax3.get_xaxis().set_tick_params(direction = 'out')
ax3.set_title('Dim')

ax31 = fig1.add_axes([0.15, 0.8, 0.5, 0.02])
cbar = fig1.colorbar(imgD, cax=ax31, orientation = 'horizontal')
cbar.ax.tick_params(labelsize = 6)
ax31.xaxis.tick_top()

ax4 = fig1.add_axes([0.4, 0.5, width, width])
imgB = ax4.imshow(np.max(normalizeImage(xlFrame['Image'].loc[combined['row'].loc['Bright'][2]]),axis=0))
ax4.axis('off')
ax3.axis('off')
imgB.set_cmap('CMRmap')
ax4.get_xaxis().set_tick_params(direction = 'out')
ax4.set_title('Bright')
outputfolder = r'C:\Users\Surf32\Desktop\ResearchDSKTOP\Miyoshi\5List\Summary'
savefile = os.path.join(outputfolder, 'img1.jpeg')
plt.savefig(savefile)



#why there are breaks in the dim line
cindex = 'Dim'
me = np.mean(combined['Above'].loc[cindex] / combined['AllPixels'].loc[cindex], axis=0)
plt.plot(me)
indexNan = np.argwhere(np.isnan(me))
indexC = indexNan[-2]
print(combined['Above'].loc[cindex][:, indexC])
print(combined['AllPixels'].loc[cindex][:, indexC])
t=combined['Above'].loc[cindex][:, indexC]/combined['AllPixels'].loc[cindex][:, indexC]
'''
#remove background from root
def removeBackground(img, dimension = 15):
    disk = skimage.morphology.square(15)
    background = np.empty(img.shape, dtype = img.dtype)
    for layer in range(img.shape[0]):
        background[layer, :, :] = skimage.morphology.opening(np.squeeze(img[layer, :, :]), disk)
    return img - background








#test HoughLinesP
plt.close('all')
#start figure
fig1 = plt.figure(figsize = [5, 5], dpi=300) 
#draw image
ax1 = fig1.add_axes([0.1, 0.1, 0.9, 0.9])
img = normalizeImage(xlFrame['Image'].loc[row])

#imgsh = ax1.imshow(np.rot90(np.max(img, axis=0)), aspect = 'equal')
imgsh = ax1.imshow(np.max(img, axis=0), aspect = 'equal')
imgsh.set_cmap('CMRmap')
ax1.get_xaxis().set_tick_params(direction = 'out')
#ax1.axis('off')
ax1.yaxis.set_visible(False)
cbar = fig1.colorbar(imgsh, location = 'top')
cbar.ax.tick_params(labelsize = 6)

import cv2

img = normalizeImage(xlFrame['Image'].loc[row])
plt.imshow(np.max(img, axis =0))
print(img.dtype)
img = np.max(img*255, axis = 0).astype(np.uint8)
plt.imshow(img)
def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    print(img.shape)
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel

testskel = skeletonize(img)
plt.imshow(testskel)







#this is where threshold is determined

#close all figures
plt.close('all')
fig1 = plt.figure(figsize = [3.5, 5], dpi=300) 
#draw image
width = 0.25
ax1 = fig1.add_axes([0.2, 0.6, width, width])
imgsh = ax1.imshow(np.max(img, axis=0), aspect = 'equal')
imgsh.set_cmap('CMRmap')
ax1.get_xaxis().set_tick_params(direction = 'out')
#ax1.axis('off')
ax1.yaxis.set_visible(False)
cbar = fig1.colorbar(imgsh, location = 'top')
cbar.ax.tick_params(labelsize = 6)

#make plots
abovethresh = img.copy()
abovethresh[img < thresh] =0
abovethresh[img >= thresh] = 1
ax2 = fig1.add_axes([0.2, 0.35, width,width])
#ax2.imshow(np.max(abovethresh, axis =0))
#ax2.plot([400, 400], [0, 500], color ='k')

#plot number of pixels above threshold
abovethreshSum=[]
for x in range(img.shape[2]):
    #ax2.plot(x, np.sum(abovethresh[:, :, x]))
    abovethreshSum.append(np.sum(abovethresh[:, :, x]))
ax2.plot(abovethreshSum, color = 'r', label = str(thresh) + ' < pixels')

#plot number of pixels below threshold
belowthresh = img.copy()
belowthresh[img < thresh] = 1
belowthresh[img >= thresh] =0
belowthresh[img < minThresh] =0
belowthreshSum = []
for x in range(img.shape[2]):
    belowthreshSum.append(np.sum(belowthresh[:, :, x]))
ax2.plot(belowthreshSum, color = 'c', label = str(minThresh) + ' < pixels < ' + str(thresh), alpha = 0.5)

#plot number of pixels above minThreshold
allPixels = img.copy()
allPixels[img >= minThresh] = 1
allPixels[img < minThresh] =0
allPixelsSum = []
for x in range(img.shape[2]):
    allPixelsSum.append(np.sum(allPixels[:, :, x]))
ax2.plot(allPixelsSum, color = 'k', label = str(minThresh) + ' < pixels ', alpha = 0.4)
ax2.set_xlim([0 , 512]) 
ax2.set_ylabel('Pixel Volume') 
#ax2.legend(loc =3, prop={'size': 6})

ax3 = fig1.add_axes([0.2, 0.05, width, width])
stdev = []
for x in range(img.shape[2]):
    stdev.append(np.std(img[:, :, x]))
ax3.plot(stdev, color = 'k', label = str(minThresh) + ' < pixels ', alpha = 0.4)
ax3.set_xlim([0 , 512]) 
ax3.set_ylabel('Stand. Deviat.')
plt.draw()
plt.savefig(os.path.join(savefolder, 'Master.png'))
'''