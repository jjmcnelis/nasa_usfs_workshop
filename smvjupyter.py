import os
import json
import warnings
from math import ceil
from io import StringIO
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from IPython.display import display

import ipyleaflet as mwg
from ipyleaflet import Map, LayerGroup, GeoJSON, CircleMarker
from ipywidgets import Layout, Button, IntProgress, Output, HBox, VBox, HTML, interactive


# ----------------------------------------------------------------------------
# app settings

#font = {"family": "normal", "weight": "normal", "size": 16}  # matplotlib ->
#plt.rcParams['figure.figsize'] = [14, 5]
#plt.rc("font", **font) # <- some matplotlib settings

warnings.filterwarnings('ignore')
auth = dict(ORNL_DAAC_USER_NUM=str(32863))
smv_download = "https://daac.ornl.gov/cgi-bin/viz/download.pl?"
smv_datasets = pd.read_csv(
    "nasa_usfs_workshop/smvdatasets.csv", 
    index_col="dataset", 
    header=0)
ignore_variables = [
    "sample","time","stat","lat","lon","FLUXNET_surface","FLUXNET_rootzone"]

# usfs shapefile fields; only applies to Yaxing's workshop -->>
fields = [
    "FORESTNUMB","DISTRICTNU","REGION","GIS_ACRES","MIN","MEDIAN","MAX",
    "RANGE","SUM","VARIETY","MINORITY","MAJORITY","COUNT"]

site_details = """
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
#{FORESTNAME} ({FORESTNUMB})
#{DISTRICTNA} ({DISTRICTNU})

# ----------------------------------------------------------------------------
# widget settings

bmap = mwg.basemap_to_tiles(mwg.basemaps.Esri.WorldImagery)    # map widget 
map_args = dict(
    center=(32.75, -109), 
    zoom=7, 
    scroll_wheel_zoom=True)

submit_args = dict(                                            # submit button
    description="Submit", 
    disabled=True, 
    button_style="success")

progress_args = dict(                                          # progress bar
    description="Progress: ", 
    layout=Layout(width="95%"))

# ----------------------------------------------------------------------------
# functions that don't support the all-in-one application


def poly_mapper(lyr, mw="25%", ow="75%", zoom=8):
    """Generates the map/plot side-by-side widget container."""

    l = (lyr["layer"].layer, lyr.points, bmap,)
    m = Map(layers=l, center=(lyr["lat"], lyr["lon"]), zoom=zoom, width=mw)
    o = Output(layout={"width": ow})
    
    return((m,o))


# ----------------------------------------------------------------------------
# SMV sample dataset formatters

numvalid = lambda v: np.count_nonzero(~np.isnan(v.data))
allnan = lambda v: numvalid(v)==0

latatts = dict(
    standard_name="latitude",
    long_name="sample latitude",
    units="degrees_north")

lonatts = dict(
    standard_name="latitude",
    long_name="sample latitude",
    units="degrees_north")

pt_style = dict(
    radius=6, 
    stroke=False,
    fill_opacity=0.6, 
    fill_color="black")


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


def pd_to_xr(dataset, df):
    """Makes an xr.Dataset from a pandas column (series) and coords."""

    a = smv_datasets.loc[dataset].to_dict()
    x = xr.DataArray(df, name=dataset, attrs=a)
    x = x.rename(dict(dim_1="stat"))
    
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
        if ("FLUXNET" not in dataset) & ("PBOH2O" not in dataset): # disabled
            split_column = split_pd(df[dataset])
            ds[dataset] = pd_to_xr(dataset, split_column)

    xds = xr.merge(ds.values())                          # merge to one xr
    xds = xds.assign_coords(lat=y, lon=x)                # add coord arrays
    
    return(xds)


class Sample(object):


    def __init__(self, i, lat, lon):
        """Inits with id,lat,lon; makes request string, map point."""
        
        self.id = i
        self.lat = lat 
        self.lon = lon

        self.on = False # on/off status user toggle
        self.pt = CircleMarker(location=(lat, lon), **pt_style)
        self.download = smv_download+"lt={0}&ln={1}&d=smap".format(lat, lon)    


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
# input polygon data

latf = "nasa_usfs_workshop/EASE2_M09km.lats.3856x1624x1.double"
lonf = "nasa_usfs_workshop/EASE2_M09km.lons.3856x1624x1.double"

lats = np.fromfile(latf, dtype=np.float64).flatten() 
lons = np.fromfile(lonf, dtype=np.float64).flatten()
crds = np.dstack((lats,lons))[0]


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


def get_layer_data(i, feat, col="#FFFFFF", opac=0.4, samp=True):
    """ """

    shapely_geom = shape(feat["geometry"])              # shapely geom
    ease = get_ease(shapely_geom) if samp else None     # ease grid points
    cent = shapely_geom.centroid                        # centroid
    lat, lon = cent.y, cent.x                           # lat, lon
    details, stats = get_properties(feat["properties"])
    feat["properties"].update({
        "id": i, 
        "style": {
            "weight": 0.75,
            "color": col,
            "fillColor": col,
            "fillOpacity": opac}})

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

"""

Make a help window!
Make hbar plot 00-50%!
Make a time line version too.



"""
# ----------------------------------------------------------------------------
# other helpers

figure_args = dict(ncols=1, sharex=True, figsize=(10,5))
bar_args = dict(stacked=True, colormap="tab20c", legend=False)
legend_ncols = lambda df: ceil(len(df.index)/10)   
download_msg = """<p style="text-align:center;">
Click <b>Submit</b> to download Soil Moisture Visualizer data for this site.
<br></p>"""


def get_ancillary_data(geojson):
    """ """
    features, cols = from_geojson(geojson)            # get features and cols

    layers = []                                       # a temporary list 
    for i, feat in enumerate(features):               # loop over features
        lyrd = get_layer_data(i, feat, opac=0, samp=False)
        lyrm = GeoJSON(
            data=lyrd[0], 
            hover_style={"color": cols[i], "fillOpacity": 0.1})
        layers.append((i, lyrd[2], lyrd[3], lyrm, lyrd[5]))
    layers = pd.DataFrame(layers, columns=["id","lat","lon","layer","attrs"])

    return(layers)


def get_nan_summary(xrdataset): 
    """ """
    nandict = {"in situ": {}, "airborne": {}, "spaceborne": {}}

    for pt in nandict.keys():

        # get the datasets for the current platform
        pds = xrdataset.filter_by_attrs(type=pt).sel(stat="Mean", drop=True)
        timelen, samplelen = pds.time.size, pds.sample.size
        potential_obs_count = timelen*samplelen

        # get variables with nodata; variables with data; valid counts
        nodata, yesdata, obscount = [], [], {}
        for name, dataset in pds.items():
            if allnan(dataset):
                nodata.append(name)
            else:
                yesdata.append(name)
                obscount[name], obstotal = [], 0
                for i in range(samplelen):
                    samp = dataset.sel(sample=i)
                    count = numvalid(samp)
                    obscount[name].append(count) #/potential_obs_count*100
                    obstotal += count
                obscount[name].append(potential_obs_count-obstotal)
        
        # update summary dictionary
        ix = list(range(samplelen))+["nan"]   
        nandict[pt].update({                      
            "nodata": nodata, 
            "yesdata": yesdata, 
            "summary": pd.DataFrame(obscount, index=ix)})

    return(nandict)


def get_nan_plot(nandict):
    """ """

    stypes = []
    for stype, nandata in nandict.items():
        cnt = len(nandata["yesdata"])
        if cnt!=0:
            stypes.append((stype, nandata["summary"], cnt))

    if len(stypes)==1:
        df = stypes[0][1]
        df.T.plot.barh(stacked=True, colormap="tab20c", figsize=(10,5))
        plt.legend(ncol=legend_ncols(df), title="sample")
    else:
        st = sorted(stypes, key=lambda x: x[2], reverse=True)
        fig, axs = plt.subplots(
            nrows=len(st), 
            gridspec_kw={'height_ratios': [i[2] for i in st]}, 
            **figure_args)
        for i, d in enumerate(st):
            d[1].T.plot.barh(ax=axs[i],**bar_args)
        fig.tight_layout()
        axs[0].legend(ncol=legend_ncols(st[0][1]), title="sample")
        axs[0].set_title("n observations by dataset")

    plt.show()


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


def get_output_layout(w="95%", b="1px solid lightgray"):
    """ """
    return({"width": w, "border": b})


class JupyterSMV(object):
    """App."""


    def __init__(self, primary=None, ancillary=None, freedom=False):

        self.polys = LayerGroup()
        self.points = LayerGroup()
        self.apolys = LayerGroup()
        self.mapw = Map(
            layers=(bmap, self.polys, self.points, self.apolys,), **map_args)

        self.submit = Button(**submit_args)
        self.submit.on_click(self.submit_handler)
        self.progress = IntProgress(**progress_args)
        
        if primary:                                   # if given, 
            self.load_features(primary)               # load input features
        if ancillary:
            self.load_ancillary(ancillary)

        layout = [self.mapw, HBox([self.submit, self.progress])]
        self.out1 = Output(layout=get_output_layout(w="80%"))
        self.out2 = Output(layout=get_output_layout(w="20%"))
        self.ui = VBox(layout + [HBox([self.out1, self.out2])])


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


    def load_ancillary(self, infeats):
        """ """          
        self.alayers = get_ancillary_data(infeats)
        for layer in self.alayers.layer:
            self.apolys.add_layer(layer)


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
        layer_row.layer.dl = True                  # set dl status to True
        
        xrds = xr.concat([s.xr for s in sample], "sample")
        lnan = get_nan_summary(xrds)
        self.layers.at[self.selected,"xr"] = xrds  # make xr dataset
        self.layers.iloc[self.selected].layer.nan = lnan

        self.out1.clear_output(); self.out2.clear_output()
        with self.out1:                            # display a summary of nan
            get_nan_plot(lnan)
        with self.out2:
            print(site_details.format(**layer_row.layer.details))


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
        self.submit.disabled = True if layer_inst.dl else False

        self.out1.clear_output(); self.out2.clear_output()
        with self.out1:                             # display nan summary
            if layer_row.layer.dl:
                get_nan_plot(layer_row.layer.nan)
            else:
                display(HTML(download_msg))
        with self.out2:
            print(site_details.format(**layer_inst.details))
