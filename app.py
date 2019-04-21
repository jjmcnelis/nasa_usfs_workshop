import os                            # core py3 
import json
from io import StringIO

import requests                      # data
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape

import ipyleaflet as mwg
from ipywidgets import Layout, Button, IntProgress, Output, HBox, VBox, HTML
from ipyleaflet import Map, LayerGroup, GeoJSON, CircleMarker
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from IPython.display import display


url = "https://daac.ornl.gov/cgi-bin/viz/download.pl?" # SMV
auth = dict(ORNL_DAAC_USER_NUM=str(32863))             # Jack
hstyle = {"color": "white", "fillOpacity": 0.6}
smvds = pd.read_csv("docs/smvdatasets.csv", index_col="dataset", header=0)

# ----------------------------------------------------------------------------
# widgets

basemap = mwg.basemap_to_tiles(mwg.basemaps.Esri.WorldImagery)
polys = LayerGroup()
points = LayerGroup()

map_center = (32.75, -109)
mapw = Map(
    layers=(basemap, polys, points,), 
    center=map_center, 
    zoom=7, 
    scroll_wheel_zoom=True)

submit = Button( 
    description='Submit', 
    disabled=True, 
    button_style='success')

progress = IntProgress(
    description="Progress: ", 
    layout=Layout(width="95%"))

ui = VBox([mapw, HBox([submit, progress])])

# ----------------------------------------------------------------------------
# ease grid

latf = "docs/EASE2_M09km.lats.3856x1624x1.double"
lonf = "docs/EASE2_M09km.lons.3856x1624x1.double"

lats = np.fromfile(latf, dtype=np.float64).flatten() 
lons = np.fromfile(lonf, dtype=np.float64).flatten()
crds = np.dstack((lats,lons))[0]
crds

def get_ease(geom):
    """ """
    bnds = geom.bounds 
    ease = crds[(bnds[1]<lats)&(lats<bnds[3])&(bnds[0]<lons)&(lons<bnds[2])]
    pt = lambda p: shape({"coordinates": p, "type": "Point"})
    inpoly = [[p[0],p[1]] for p in ease if geom.contains(pt([p[1], p[0]]))]
    return(inpoly)


# ----------------------------------------------------------------------------
# SMV dataset formatters


def txt_to_pd(response_text):
    """Parses response.text to data frame with date index."""
    
    f = StringIO(response_text)                      # get file from string
    df = pd.read_csv(f, header=4, index_col="time")  # read to df
    df.index = pd.to_datetime(df.index)              # convert index to dates
    
    return(df)


def split_pd(col):
    """Splits pd column by ; and set all values to float, nan."""
    
    df = col.str.split(";",n=2,expand=True)           # split col by ;
    df = df.replace('', np.nan)                       # set '' to nan
    df = df.astype(float)                             # set all to float
    df.columns = ["mean","min","max"]                 # add column names
    
    return(df)


# ----------------------------------------------------------------------------
# xarray 

latatts = dict(
    standard_name="latitude",
    long_name="sample latitude",
    units="degrees_north")

lonatts = dict(
    standard_name="latitude",
    long_name="sample latitude",
    units="degrees_north")


def pd_to_xr(dataset, df):
    """Makes an xr.Dataset from a pandas column (series) and coords."""
    
    a = smvds.loc[dataset].to_dict()
    x = xr.DataArray(df, name=dataset, attrs=a)
    x = x.rename(dict(dim_1="stat"))
    x.attrs["allnan"] = int(np.isnan(np.nanmean(x.data)))
    
    return(x)


def get_sample_xr(samp):
    """ """

    d = ["sample"]                          # get sample, lat, lon xr arrays
    s = xr.DataArray(data=[samp.id], dims=d)
    y = xr.DataArray(data=[samp.lat], coords=[s], dims=d, attrs=latatts)
    x = xr.DataArray(data=[samp.lon], coords=[s], dims=d, attrs=lonatts)

    df = samp.df                                         # get the sample df
    dfs = {col: split_pd(df[col]) for col in df.columns} # split cols to dfs
    ds = {c: pd_to_xr(c,d) for c,d in dfs.items()}       # make xr datasets
    xds = xr.merge(ds.values())                          # merge to one xr
    xds = xds.assign_coords(lat=y, lon=x)                # add coord arrays
    
    return(xds)


# ----------------------------------------------------------------------------
url = "https://daac.ornl.gov/cgi-bin/viz/download.pl?"

pt_style = dict(radius=7, fill_opacity=0.6, fill_color="black", stroke=False)
pt_status_on = dict(stroke=True, color="white", opacity=0.6)
pt_status_off = dict(stroke=False, color="black", opacity=0.6)


class Sample(object):

    def __init__(self, i, lat, lon):
        """Inits with id,lat,lon; makes request string, map point."""
        self.id, self.lat, self.lon = i, lat, lon
        self.rurl = url+"lt={0}&ln={1}&d=smap".format(lat,lon)  # request url
        self.pt = CircleMarker(location=(lat, lon), **pt_style) # map point
        self.on = False                                         # status

    def update(self, **kwargs):
        for arg, val in kwargs.items():
            setattr(self.pt, arg, val)
    
    def toggle(self, event, type, coordinates):
        opac = 0.1 if self.on else 0.6
        self.update(opacity=opac)
        self.on = False if self.on else True
        
    def submit(self):
        """Called by parent. Downloads url. Updates status."""
        self.response = requests.get(self.rurl, cookies=auth)  # submit SMV 
        self.df = txt_to_pd(self.response.text)                # read to df
        self.xr = get_sample_xr(self)                          # get xr
        self.pt.on_click(self.toggle)                          # callback
        self.on = True                                         # toggle on


# ----------------------------------------------------------------------------
# app classes

lyr_style = lambda c: {"color":c,"fillColor":c,"weight":1,"fillOpacity": 0.4}
lyr_hstyle = {"color": "white", "fillOpacity": 0.8}
statcheck = lambda k: k.split("_")[0] not in ["MEAN","STD","Count", "style"]


def mgeo(i, feat, col):
    feat["properties"].update({
        "id": i, 
        "style": lyr_style(col)})
    return(dict(data=feat, hover_style=lyr_hstyle))


class Layer(object):

    def __init__(self, i, feat, col=None):
        """Inits with id,lat,lon; makes request string, map point."""
        self.id = i
        self.feat = feat
        
        self.sgeom = shape(feat["geometry"])
        self.ease = get_ease(self.sgeom)                    # get ease points
        self.cent = self.sgeom.centroid                     # get centroid
        self.lat, self.lon = self.cent.y, self.cent.x       # get lat, lon
        
        prop = feat["properties"]
        mean = [v for k,v in prop.items() if "MEAN" in k]
        std = [v for k,v in prop.items() if "STD" in k]
        self.stats = pd.DataFrame({"mean": mean, "std": std})
        self.site = {k:v for k,v in prop.items() if statcheck(k)}
        
        lyr = mgeo(i, feat, col)
        self.layer = GeoJSON(**lyr)
        self.layer.on_click(self.toggle)

        self.on, self.dl = False, False                    # on/off, dl status
        
    def update(self, **kwargs):
        for arg, val in kwargs.items():
            setattr(self.layer, arg, val)
    
    def toggle(self, **kwargs):
        """Routine for when a new USFS polygon is selected."""
        if list(kwargs.keys()) != ['event', 'properties']: # check event
            return(None)                                   # skip basemap
        self.on = False if self.on else True               # update status


# ----------------------------------------------------------------------------
# callbacks

prog = lambda m: dict(min=0, max=m, value=0)
xrds = lambda l: xr.concat([s.xr for s in l], "sample")


def get_on_lyrs(column=None):
    """
    Returns a subset of layers df, only "on" layers. If keyword 
    argument 'column' will return only that column.
    """
    on = [i for i,row in layers.iterrows() if row["layer"].on]
    sdf = layers.iloc[on][column] if column else layers.iloc[on]
    return(sdf)


def submit_handler(b):
    """Resets UI and sends requests to SMV when new submit."""
    
    lyron = get_on_lyrs()
    for i, row in lyron.iterrows():             # loop over samples col
        if not row["layer"].dl:                 # if not downloaded yet
            samp = row.samples["samp"].tolist() # get samples
            for a,v in prog(len(samp)).items(): # reset progress bar
                setattr(progress, a, v)
            for s in samp:                      # loop over sample pts
                progress.value += 1             # update progress bar
                s.update(**pt_status_on)        # update style
                s.submit()                      # download the data
            layers.at[i,"xr"] = xrds(samp)      # make xr dataset
            row["layer"].dl = True              # set dl status to True
    submit.disabled = True                      # disable submit button

submit.on_click(submit_handler)


def layer_click_handler(**kwargs): 
    """
    Routine for when a new USFS polygon is selected. Layer.toggle
    internal updater should evaluate first.
    """
    if list(kwargs.keys()) != ['event', 'properties']: # check event
         return(None)                         # skip basemap
    i = int(kwargs["properties"]["id"])       # set selected poly id
    l = layers.iloc[i]                        # get row for selected
    lo = l.layer                              # get Layer class inst
    pteval = points.add_layer if lo.on else points.remove_layer
    pteval(l["points"])                       # update layer status;
    submit.disabled = True if len(get_on_lyrs())==0 else False
    if lo.on:
        mapw.center, mapw.zoom = (lo.lat,lo.lon), 9 # ctr,zoom map


# ----------------------------------------------------------------------------
# data prep

with open("sites/Sites_lf_geo.json", "r") as f:
    shapes = json.load(f)

features = shapes["features"]

cspace = np.linspace(0.0, 1.0, len(features)) # 1
rgb = cm.Set3(cspace)                         # 2
cols = [colors.to_hex(c[0:3]) for c in rgb]   # 3

sample_header = ["id","lat","lon","samp"]
layer_header = ["id","lat","lon","layer","samples","points","xr"]

layers = []                                   # a temporary list 
for i, feat in enumerate(features):           # loops over USFS poly feats
    
    poly = Layer(i, feat, cols[i])            # get Layer class
    poly.layer.on_click(layer_click_handler)  # set global callback
    polys.add_layer(poly.layer)               # add to polys layer group

    pts, samps = LayerGroup(), []             # points layer group; Samples
    for j, p in enumerate(poly.ease):         # loop over EASE grid pts
        s = Sample(j, p[0], p[1])             # make a Sample instance
        pts.add_layer(s.pt)                   # add to points layer group
        samps.append((j, p[0], p[1], s))      # append tuple to the list  
        
    samps = pd.DataFrame(samps, columns=sample_header)       # samples df
    layers.append((i,poly.lat,poly.lon,poly,samps,pts,None)) # append
    
layers = pd.DataFrame(layers, columns=layer_header)          # layers df
