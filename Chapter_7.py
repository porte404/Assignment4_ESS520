
# coding: utf-8

# In[23]:

import numpy as np
import linecache as lc

myArray1 = np.loadtxt("myGrid.asc", skiprows=6)
line1 = lc.getline("myGrid.asc", 1)
line2 = lc.getline("myGrid.asc", 2)
line3 = lc.getline("myGrid.asc", 3)
line4 = lc.getline("myGrid.asc", 4)
line5 = lc.getline("myGrid.asc", 5)
line6 = lc.getline("myGrid.asc", 6)

myArray = myArray1.copy()


# In[24]:

header = line1
header += line2
header += line3
header += line4
header += line5
header += line6

np.savetxt("myGrid_2.asc", myArray, header=header, fmt="%1.2f")
myArray2 = np.loadtxt("myGrid.asc", skiprows=6)
myArray2


# In[30]:

# File name of ASCII digital elevation model
source = "dem.asc"
# File name of the slope grid
slopegrid = "slope.asc"
# File name of the aspect grid
aspectgrid = "aspect.asc"
# Output file name for shaded relief
shadegrid = "relief.asc"
## Shaded elevation parameters
# Sun direction
azimuth=315.0
# Sun angle
altitude=45.0
# Elevation exageration
z=1.0
# Resolution
scale=1.0
# No data value for output
NODATA = -9999
# Needed for numpy conversions
deg2rad = 3.141592653589793 / 180.0
rad2deg = 180.0 / 3.141592653589793
# Parse the header using a loop and
# the built-in linecache module
hdr = [lc.getline(source, i) for i in range(1,7)]
values = [float(h.split(" ")[-1].strip())  for h in hdr]
cols,rows,lx,ly,cell,nd = values
xres = cell
yres = cell * -1
# Load the dem into a numpy array
arr = np.loadtxt(source, skiprows=6)


# In[31]:

# Exclude 2 pixels around the edges which are usually NODATA.
# Also set up structure for a 3x3 window to process the slope
# throughout the grid
window = []
for row in range(3):
    for col in range(3):
        window.append(arr[row:(row + arr.shape[0] - 2),                           col:(col + arr.shape[1] - 2)])
# Process each cell
x = ((z * window[0] + z * window[3] + z *       window[3] + z * window[6]) -      (z * window[2] + z * window[5] + z *       window[5] + z * window[8])) / (8.0 * xres * scale);
y = ((z * window[6] + z * window[7] + z * window[7] + z * window[8])      - (z * window[0] + z * window[1] + z * window[1] + z * window[2]))     / (8.0 * yres * scale);
# Calculate slope
slope = 90.0 - np.arctan(np.sqrt(x*x + y*y)) * rad2deg
# Calculate aspect
aspect = np.arctan2(x, y)
# Calculate the shaded relief
shaded = np.sin(altitude * deg2rad) * np.sin(slope * deg2rad) + np.cos(altitude * deg2rad) * np.cos(slope * deg2rad) * np.cos((azimuth - 90.0) * deg2rad - aspect);
shaded = shaded * 255
# Rebuild the new header
header = "ncols %s\n" % shaded.shape[1]
header += "nrows %s\n" % shaded.shape[0]
header += "xllcorner %s\n" % (lx + (cell * (cols - shaded.shape[1])))
header += "yllcorner %s\n" % (ly + (cell * (rows - shaded.shape[0])))
header += "cellsize %s\n" % cell
header += "NODATA_value %s\n" % NODATA
# Set no-data values
for pane in window:
    slope[pane == nd] = NODATA
    aspect[pane == nd] = NODATA
    shaded[pane == nd] = NODATA


# In[35]:

# Open the output file, add the header, save the slope grid (THIS IS IN PYTHON 3)
with open(slopegrid, "wb") as f:
    np.savetxt(f, slope, header=header, fmt="%4i")
    
# Open the output file, add the header, save the slope grid
with open(aspectgrid, "wb") as f:
    np.savetxt(f, aspect, header=header, fmt="%4i")

# Open the output file, add the header, save the array
with open(shadegrid, "wb") as f:
    np.savetxt(f, shaded, header=header, fmt="%4i")


# In[36]:

import ogr
import gdal


# In[41]:

# Elevation DEM
source = "dem.asc"
# Output shapefile
target = "contour"
ogr_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(target + ".shp")
ogr_lyr = ogr_ds.CreateLayer(target, geom_type = ogr.wkbLineString25D)
field_defn = ogr.FieldDefn('ID', ogr.OFTInteger)
ogr_lyr.CreateField(field_defn)
field_defn = ogr.FieldDefn('ELEV', ogr.OFTReal)
ogr_lyr.CreateField(field_defn)
# gdal.ContourGenerate() arguments
# Band srcBand,
# double contourInterval,
# double contourBase,
# double[] fixedLevelCount,
# int useNoData,
# double noDataValue,
# Layer dstLayer,
# int idField,
# int elevField
ds = gdal.Open('dem.asc')
gdal.ContourGenerate(ds.GetRasterBand(1), 400, 10, [], 0, 0, ogr_lyr, 0, 1)
del ogr_ds


# In[48]:

import shapefile
import pngcanvas
# Open the contours
r = shapefile.Reader("contour.shp")
# Setup the world to pixels conversion
xdist = r.bbox[2] - r.bbox[0]
ydist = r.bbox[3] - r.bbox[1]
iwidth = 800
iheight = 600
xratio = iwidth/xdist
yratio = iheight/ydist
contours = []
# Loop through all shapes
for shape in r.shapes():
    # Loop through all parts
    for i in range(len(shape.parts)):
        pixels=[]
        pt = None
        if i<len(shape.parts)-1:
            pt = shape.points[shape.parts[i]:shape.parts[i+1]]
        else:
            pt = shape.points[shape.parts[i]:]
        for x,y in pt:
            px = int(iwidth - ((r.bbox[2] - x) * xratio))
            py = int((r.bbox[3] - y) * yratio)
            pixels.append([px,py])
        contours.append(pixels)
# Set up the output canvas
canvas = pngcanvas.PNGCanvas(iwidth,iheight)
# PNGCanvas accepts rgba byte arrays for colors
red = [0xff,0,0,0xff]
canvas.color = red
# Loop through the polygons and draw them
for c in contours:
    canvas.polyline(c)
# Save the image
f = open("contours.png", "wb")
f.write(canvas.dump())
f.close()


# In[49]:

from laspy.file import File
import numpy as np

# Source LAS file
source = "lidar.las"
# Output ASCII DEM file
target = "lidar.asc"
# Grid cell size (data units)
cell = 1.0
# No data value for output DEM
NODATA = 0
# Open LIDAR LAS file
las = File(source, mode="r")
#xyz min and max
min = las.header.min
max = las.header.max
# Get the x axis distance
xdist = max[0] - min[0]
# Get the y axis distance
ydist = max[1] - min[1]
# Number of columns for our grid
cols = int(xdist) / cell
# Number of rows for our grid
rows = int(ydist) / cell
cols += 1
rows += 1
# Track how many elevation
# values we aggregate
count = np.zeros((rows, cols)).astype(np.float32)
# Aggregate elevation values
zsum = np.zeros((rows, cols)).astype(np.float32)
# Y resolution is negative
ycell = -1 * cell
# Project x,y values to grid
projx = (las.x - min[0]) / cell
projy = (las.y - min[1]) / ycell
# Cast to integers and clip for use as index
ix = projx.astype(np.int32)
iy = projy.astype(np.int32)
# Loop through x,y,z arrays, add to grid shape,
# and aggregate values for averaging
for x,y,z in np.nditer([ix, iy, las.z]):
    count[y, x]+=1
    zsum[y, x]+=z

# Change 0 values to 1 to avoid numpy warnings,
# and NaN values in array
nonzero = np.where(count>0, count, 1)
# Average our z values
zavg = zsum/nonzero
# Interpolate 0 values in array to avoid any
# holes in the grid
mean = np.ones((rows,cols)) * np.mean(zavg)
left = np.roll(zavg, -1, 1)
lavg = np.where(left>0,left,mean)
right = np.roll(zavg, 1, 1)
ravg = np.where(right>0,right,mean)
interpolate = (lavg+ravg)/2
fill=np.where(zavg>0,zavg,interpolate)
# Create our ASCII DEM header
header = "ncols %s\n" % fill.shape[1]
header += "nrows %s\n" % fill.shape[0]
header += "xllcorner %s\n" % min[0]
header += "yllcorner %s\n" % min[1]
header += "cellsize %s\n" % cell
header += "NODATA_value %s\n" % NODATA
# Open the output file, add the header, save the array
with open(target, "wb") as f:
    # The fmt string ensures we output floats
    # that have at least one number but only
    # two decimal places
    np.savetxt(f, fill, header=header, fmt="%1.2f")


# In[53]:

import numpy as np
from PIL import Image
from PIL import ImageOps
# Source LAS file
source = "relief.asc"
# Output ASCII DEM file
target = "relief.bmp"
# Load the ASCII DEM into a numpy array
arr = np.loadtxt(source, skiprows=6)
# Convert array to numpy image
im = Image.fromarray(arr).convert('RGB')
# Enhance the image:
# equalize and increase contrast
im = ImageOps.equalize(im)
im = ImageOps.autocontrast(im)
# Save the image
im.save(target)


# In[51]:

import colorsys


# In[52]:

# Source LIDAR file
source = "lidar.asc"
# Output image file
target = "lidar.bmp"
# Load the ASCII DEM into a numpy array
arr = np.loadtxt(source, skiprows=6)
# Convert the numpy array to a PIL image
im = Image.fromarray(arr).convert('L')
# Enhance the image
im = ImageOps.equalize(im)
im = ImageOps.autocontrast(im)
# Begin building our color ramp
palette = []
# Hue, Saturaction, Value
# color space
h = .67
s = 1
v = 1
# We'll step through colors from:
# blue-green-yellow-orange-red.
# Blue=low elevation, Red=high-elevation
step = h/256.0
# Build the palette
for i in range(256):
    rp,gp,bp = colorsys.hsv_to_rgb(h,s,v)
    r = int(rp*255)
    g = int(gp*255)
    b = int(bp*255)
    palette.extend([r,g,b])
    h-=step
# Apply the palette to the image
im.putpalette(palette)
# Save the image
im.save(target)


# In[58]:

#import cPickle
import os
import time
import math
# Third-party Python modules:
import numpy as np
import shapefile
from laspy.file import File
#import voronoi

#I am having problems with this section because I am using python3. The voronoi.py file has too many syntax errors
#and cPickle is only python 2 compatable


# In[ ]:



