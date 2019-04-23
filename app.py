import os                            # core py3 
import json
import warnings
from io import StringIO

import requests                      # data
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape

import ipyleaflet as mwg
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from IPython.display import display
from ipyleaflet import Map, LayerGroup, GeoJSON, CircleMarker
from ipywidgets import Layout, Button, IntProgress, Output, HBox, VBox, HTML, interactive

fields = [                               # usfs shapefile fields - delete
    "FORESTNUMB",
    "DISTRICTNU",
    "REGION",
    "GIS_ACRES",
	"MIN",
	"MEDIAN",
	"MAX",
	"RANGE",
	"SUM",
	"VARIETY",
	"MINORITY",
	"MAJORITY",
	"COUNT"
]

site_details = """
{FORESTNAME} ({FORESTNUMB})
{DISTRICTNA} ({DISTRICTNU})
REGION:   {REGION}
ACRES:    {GIS_ACRES}
MIN:      {MIN}
MEDIAN:   {MEDIAN}
MAX:      {MAX}
RANGE:    {RANGE}
SUM:      {SUM}
VARIETY:  {VARIETY}
MINORITY: {MINORITY}
MAJORITY: {MAJORITY}
COUNT:    {COUNT}
"""


# ----------------------------------------------------------------------------
# app settings

warnings.filterwarnings('ignore')
auth = dict(ORNL_DAAC_USER_NUM=str(32863))
smv_download = "https://daac.ornl.gov/cgi-bin/viz/download.pl?"
smv_datasets = pd.read_csv(
    "docs/smvdatasets.csv", 
    index_col="dataset", 
    header=0)

# ----------------------------------------------------------------------------
# widget settings

# map widget 
bmap = mwg.basemap_to_tiles(mwg.basemaps.Esri.WorldImagery)
map_args = dict(
    center=(32.75, -109), 
    zoom=7, 
    scroll_wheel_zoom=True)

# submit button
submit_args = dict(
    description="Submit", 
    disabled=True, 
    button_style="success")

# progress bar
progress_args = dict(
    description="Progress: ", 
    layout=Layout(width="95%"))

# ----------------------------------------------------------------------------
# ease grid

latf = "docs/EASE2_M09km.lats.3856x1624x1.double"
lonf = "docs/EASE2_M09km.lons.3856x1624x1.double"

lats = np.fromfile(latf, dtype=np.float64).flatten() 
lons = np.fromfile(lonf, dtype=np.float64).flatten()
crds = np.dstack((lats,lons))[0]


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
    df.columns = ["Max","Mean","Min"]                 # add column names
    
    return(df)


# ----------------------------------------------------------------------------
# xarray 

allnan = lambda v: np.count_nonzero(~np.isnan(v.data))==0

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

    a = smv_datasets.loc[dataset].to_dict()

    x = xr.DataArray(df, name=dataset, attrs=a)
    x = x.rename(dict(dim_1="stat"))
    x.attrs["mean_nan"] = int(allnan(x.sel(stat="Mean")))
    x.attrs["min_nan"] = int(allnan(x.sel(stat="Min")))
    x.attrs["max_nan"] = int(allnan(x.sel(stat="Max")))
    
    return(x)


def get_sample_xr(samp):
    """ """

    d = ["sample"]                          
    s = xr.DataArray(data=[samp.id], dims=d) # get sample, lat, lon xr arrays
    y = xr.DataArray(data=[samp.lat], coords=[s], dims=d, attrs=latatts)
    x = xr.DataArray(data=[samp.lon], coords=[s], dims=d, attrs=lonatts)

    df = samp.df                                         # get the sample df
    ds = {}
    for dataset in df.columns:
        if "FLUXNET" not in dataset:
            split_column = split_pd(df[dataset])
            ds[dataset] = pd_to_xr(dataset, split_column)

    xds = xr.merge(ds.values())                          # merge to one xr
    xds = xds.assign_coords(lat=y, lon=x)                # add coord arrays
    
    return(xds)


# ----------------------------------------------------------------------------
url = "https://daac.ornl.gov/cgi-bin/viz/download.pl?"

pt_style = dict(
    radius=7, 
    stroke=False,
    fill_opacity=0.6, 
    fill_color="black")


class Sample(object):


    def __init__(self, i, lat, lon):
        """Inits with id,lat,lon; makes request string, map point."""
        
        self.id = i
        self.lat = lat 
        self.lon = lon

        self.on = False # on/off status user toggle
        self.pt = CircleMarker(location=(lat, lon), **pt_style)
        self.download = url + "lt={0}&ln={1}&d=smap".format(lat, lon)    


    def update(self, **kwargs):
        for arg, val in kwargs.items():
            setattr(self.pt, arg, val)


    def toggle(self, event, type, coordinates):
        
        opac = 0.1 if self.on else 0.6             # determine opacity
        self.update(opacity=opac)                  # set opacity
        self.on = False if self.on else True       # toggle status


    def submit(self):
        """Called by parent. Downloads url. Updates status."""
        
        self.response = requests.get(self.download, cookies=auth) # dl SMV 
        self.df = txt_to_pd(self.response.text)    # read to df
        self.xr = get_sample_xr(self)              # get xr dataset
        self.pt.on_click(self.toggle)              # callback switch style
        self.on = True                             # toggle status on


# ----------------------------------------------------------------------------
# usfs data


def get_ease(shapely_geom):
    """ """

    bnds = shapely_geom.bounds 
    ease = crds[
        (bnds[1]<lats) & (lats<bnds[3]) &     # ybnds < lat < ybnds
        (bnds[0]<lons) & (lons<bnds[2])]      # xbnds < lon < xbnds

    ease_reduced = []
    for p in ease:
        shapely_pt = shape({                  # input to shapely.shape is a
            "type": "Point",                  # python dict equivalent of
            "coordinates": (p[1], p[0])})     # geojson point geometry
        
        if shapely_geom.contains(shapely_pt): # if point inside poly
            ease_reduced.append([p[0], p[1]]) # return lat, lon tuple

    return(ease_reduced)

def get_properties(prop):
    """ """

    details, stats = {}, {"MEAN": [], "STD": []}
    for key, value in prop.items():
        if key in fields:
            details[key] = value
        elif "MEAN_" in key:
            stats["MEAN"].append(value)
        elif "STD_" in key:
            stats["STD"].append(value)
        else:
            pass
    yr = pd.date_range(
        start="1985",
        freq="1Y",
        periods=len(stats["MEAN"]))
    stats = pd.DataFrame(stats, index=yr)

    return((details, stats))

def get_layer_data(i, feat, col, fields=["FID"]):
    """ """

    shapely_geom = shape(feat["geometry"])    # shapely geom
    ease = get_ease(shapely_geom)             # ease grid points
    cent = shapely_geom.centroid              # centroid
    lat = cent.y                              # lat, lon
    lon = cent.x
    details, stats = get_properties(feat["properties"])
    feat["properties"].update({
        "id": i, 
        "style": {
            "weight": 1,
            "color": col,
            "fillColor": col,
            "fillOpacity": 0.4}})

    return((feat, ease, lat, lon, stats, details))


class Layer(object):


    def __init__(self, i, feat, col=None):
        """Inits with id,lat,lon; makes request string, map point."""
        
        layer_data = get_layer_data(i, feat, col)
        
        self.id = i
        self.feat = layer_data[0]
        self.ease = layer_data[1]
        self.lat = layer_data[2]
        self.lon = layer_data[3]
        self.stats = layer_data[4]
        self.details = layer_data[5]

        self.layer = GeoJSON(
            data=self.feat,
            hover_style={
                "color": "white", 
                "fillOpacity": 0.8})
        self.layer.on_click(self.toggle)

        self.dl = False    # downloaded or nah?
        self.on = False    # toggle on or nah?


    def toggle(self, **kwargs):
        """Routine for when a new USFS polygon is selected."""

        if list(kwargs.keys()) != ['event', 'properties']: # check event
            return(None)                                   # skip basemap
        
        self.on = False if self.on else True               # update status
        

    def update(self, **kwargs):
        for arg, val in kwargs.items():
            setattr(self.layer, arg, val)


# ----------------------------------------------------------------------------
# app

pt_status_on = dict(
    opacity=0.5,
    stroke=True, 
    color="white")

pt_status_off = dict(
    opacity=0.6,
    stroke=False, 
    color="black")

sample_header = [
    "id",
    "lat",
    "lon",
    "samp"
]

layer_header = [
    "id",
    "lat",
    "lon",
    "layer",
    "samples",
    "points",
    "xr"
]

ignorevars = ["time","stat","sample","lat","lon"]


def get_colors(n, cmap=cm.Set3):
    """ """

    cspace = np.linspace(0.0, 1.0, n)           # 1
    rgb = cmap(cspace)                          # 2
    cols = [colors.to_hex(c[0:3]) for c in rgb] # 3

    return(cols)

def from_geojson(input_geojson):
    """ """
    with open(input_geojson, "r") as f:
        shapes = json.load(f)
    features = shapes["features"]
    cols = get_colors(len(features))
    return((features, cols))

def get_plottable(xds):
    """ """
    plotvars = []
    for v in list(xds.variables):
        if v not in ignorevars:
            if not allnan(xds[v]):
                plotvars.append(v)
    return(plotvars)

def get_active_samples(layer):
    """ """
    on = []
    for sample in layer.samples.samp:
        if sample.on:
            on.append(sample.id)
    return(on)


class JupyterSMV(object):
    """App."""


    def __init__(self, in_features=None, mpl_notebook=False):

        self.polys = LayerGroup()
        self.points = LayerGroup()
        self.mapw = Map(layers=(bmap, self.polys, self.points,), **map_args)

        self.submit = Button(**submit_args)
        self.submit.on_click(self.submit_handler)
        self.progress = IntProgress(**progress_args)

        layout = [self.mapw, HBox([self.submit, self.progress])]

        if mpl_notebook:
            self.init_plotter = self.live_plot
            self.fig, self.axs = plt.subplots(3,1)
        else:
            self.init_plotter = self.static_plot
            self.out = Output()
            self.out.layout = {"width": "95%"}
            layout = layout + [self.out]
        
        if in_features:                               # if given, 
            self.load_features(in_features)           # load input features

        self.ui = VBox(layout)


    def load_features(self, infeats):
        """ """
        features, cols = from_geojson(infeats)        # get features and cols

        layers = []                                   # a temporary list 
        for i, feat in enumerate(features):           # loop over features
            
            poly = Layer(i, feat, cols[i])            # get Layer class
            poly.layer.on_click(self.layer_click_handler)  # global callback
            self.polys.add_layer(poly.layer)          # add to poly grp

            pts, samps = LayerGroup(), []             # points group; Samples
            for j, p in enumerate(poly.ease):         # loop EASE grid pts
                s = Sample(j, p[0], p[1])             # make Sample instance
                pts.add_layer(s.pt)                   # add to points group
                samps.append((j, p[0], p[1], s))      # append tuple to list  

            samps = pd.DataFrame(samps, columns=sample_header)       # samples
            layers.append((i,poly.lat,poly.lon,poly,samps,pts,None)) # append
        
        self.layers = pd.DataFrame(layers, columns=layer_header)     # layers
        self.selected = None


    def submit_handler(self, b):
        """Resets UI and sends requests to SMV when new submit."""
        
        layer_row = self.layers.iloc[self.selected]
        sample = layer_row.samples["samp"].tolist()

        self.progress.min = 0                      # reset progress bar
        self.progress.max = len(sample)
        self.progress.value = 0

        for s in sample:                           # loop over sample pts
            self.progress.value += 1               # update progress bar
            s.update(**pt_status_on)               # update style
            s.submit()                             # download the data
            s.pt.on_click(self.init_plotter)
        xrds = xr.concat([s.xr for s in sample], "sample")
        self.layers.at[self.selected,"xr"] = xrds  # make xr dataset

        layer_row.layer.dl = True                  # set dl status to True
        self.init_plotter()


    def layer_click_handler(self, **kwargs): 
        """ Evaluates when new polygon is selected. Layer.toggle first!"""
        
        if list(kwargs.keys()) != ['event', 'properties']: # check event
            return(None)                                   # skip basemap

        i = int(kwargs["properties"]["id"])             # get selected poly id
        layer_row = self.layers.iloc[i]                 # get row for selected
        layer_inst = layer_row.layer                    # get Layer class inst
        self.selected = i

        self.points.clear_layers()
        self.points.add_layer(layer_row["points"]) 
        self.mapw.center = (layer_inst.lat, layer_inst.lon)
        self.mapw.zoom = 9

        if layer_inst.dl:
            self.init_plotter()
            self.submit.disabled = True
        else:
            self.submit.disabled = False

    # ------------------------------------------------------------------------

    def static_plot(self, event=None, type=None, coordinates=None):
        """ """

        lyr = self.layers.iloc[self.selected]     
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(11, 8))
        xds = lyr.xr 

        # xds dimension filter
        active_samples = get_active_samples(lyr)
        dimension_filter = dict(stat="Mean", sample=active_samples)
        xdsf = xds.sel(dimension_filter)

        # get plottable variables
        plottable = get_plottable(xdsf)
        xdsf = xdsf[plottable]

        # USFS productivity statistics ---------------------------------------
        
        st, et = xdsf.time.data[0], xdsf.time.data[-1]   # xds time bounds

        stats = lyr.layer.stats.loc[st:et]
        stats.MEAN.plot(color="black", ax=axs[0])
        (stats.MEAN-stats.STD).plot(color="black", ls=":", ax=axs[0])
        (stats.MEAN+stats.STD).plot(color="black", ls=":", ax=axs[0])

        # SMV datasets -------------------------------------------------------

        # get surface and rootzone
        xdsf_surf = xdsf.filter_by_attrs(soil_zone="surface")
        for d in xdsf_surf:
            xdsf_surf[d].mean("sample").plot.line(x='time', ax=axs[1])

        xdsf_root = xdsf.filter_by_attrs(soil_zone="rootzone")
        for d in xdsf_root:
            xdsf_root[d].mean("sample").plot.line(x='time', ax=axs[2])

        # draw ---------------------------------------------------------------

        fig.tight_layout()
        self.out.clear_output()
        with self.out:
            plt.show()


    def live_plot(self):
        lyr = self.layers.iloc[self.selected]
        xds = lyr.xr
        plotvars = get_plottable(xds)

        mean = lyr.layer.stats.GPP_mean                        # USFS -->> 
        std = lyr.layer.stats.GPP_std
        mean.plot(color="black", ax=self.axs[2])
        (mean-std).plot(color="black", ls=":", ax=self.axs[2])
        (mean+std).plot(color="black", ls=":", ax=self.axs[2]) # <<-- USFS

        def update(Dataset, Statistic):
            """ """
            Samples = get_active_samples(lyr)
            select = dict(stat=Statistic, sample=Samples)
            self.axs[0].clear(); self.axs[1].clear(); self.axs[2].clear()
            xds[Dataset].sel(select).plot.line(x='time', ax=self.axs[0])
            xds[Dataset].sel(select).mean("sample").plot.line(x='time', ax=self.axs[1])
            self.fig.canvas.draw()

        widgets = dict(Dataset=plotvars, Statistic=["Mean","Min","Max"])
        p = interactive(update, **widgets)
        display(p)
