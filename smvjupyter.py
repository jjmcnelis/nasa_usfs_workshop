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
from ipywidgets import RadioButtons, Box, IntSlider, Dropdown, ToggleButton, GridBox
from ipywidgets import Layout, Button, IntProgress, Output, HBox, VBox, HTML, interactive, SelectionRangeSlider


disabled_sources =  ["FLUXNET", "MODIS", "GRACE", "PBOH2O"]
ignore_variables = ["sample","time","stat","lat","lon"]

# ----------------------------------------------------------------------------
# app settings

plt.ioff()
plt.clf()
font = {'family': 'normal', 'weight': 'normal', 'size': 8}
plt.rc('font', **font)

warnings.filterwarnings('ignore')

smvdownload = "https://airmoss.ornl.gov/cgi-bin/viz/api/download.pl?"
smvdatasets = pd.read_csv(
    "docs/smvdatasets.csv", 
    index_col="dataset", 
    header=0)

# usfs shapefile fields; only applies to Yaxing's workshop -->>
fields = [
    "FORESTNAME","FORESTNUMB","DISTRICTNA","DISTRICTNU","REGION","GIS_ACRES","MIN","MEDIAN","MAX",
    "RANGE","SUM","VARIETY","MINORITY","MAJORITY","COUNT"]

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

# ease data
latf = "docs/EASE2_M09km.lats.3856x1624x1.double"
lonf = "docs/EASE2_M09km.lons.3856x1624x1.double"

lats = np.fromfile(latf, dtype=np.float64).flatten() 
lons = np.fromfile(lonf, dtype=np.float64).flatten()
crds = np.dstack((lats,lons))[0]

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

figure_args = dict(ncols=1, sharex=True, figsize=(10,5))
bar_args = dict(stacked=True, colormap="tab20c", legend=False)
legend_ncols = lambda df: ceil(len(df.index)/10)   
download_msg = """<p style="text-align:center;">
Click <b>Submit</b> to download Soil Moisture Visualizer data for this site.
<br></p>"""


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


def poly_mapper(lyr, mw="25%", ow="75%", zoom=8):
    """Generates the map/plot side-by-side widget container."""

    l = (lyr["layer"].layer, lyr.points, bmap,)
    m = Map(layers=l, center=(lyr["lat"], lyr["lon"]), zoom=zoom, width=mw)
    o = Output(layout={"width": ow})
    
    return((m,o))


def get_colors(n, cmap=cm.Set2):
    """ 
    Takes integer count of colors to map and optional kwarg: 
      'cmap=matplotlib.cm.<cmap>'.
    """

    cspace = np.linspace(0.0, 1.0, n)           # *1
    rgb = cmap(cspace)                          # *2
    cols = [colors.to_hex(c[0:3]) for c in rgb] # *3

    return(cols)


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


# ----------------------------------------------------------------------------
# SMV sample dataset formatters


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
    df.columns = ["Min","Mean","Max"]                 # add column names
    
    return(df)


def pd_to_xr(dataset, df):
    """Makes an xr.Dataset from a pandas column (series) and coords."""

    a = smvdatasets.loc[dataset].to_dict()
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
        a = smvdatasets.loc[dataset].to_dict()
        if a["source"] not in disabled_sources:
            split_column = split_pd(df[dataset])
            ds[dataset] = pd_to_xr(dataset, split_column)

    xds = xr.merge(ds.values())                          # merge to one xr
    xds = xds.assign_coords(lat=y, lon=x)                # add coord arrays
    
    return(xds)


def from_geojson(input_geojson):
    """ """
    with open(input_geojson, "r") as f:
        shapes = json.load(f)
    features = shapes["features"]
    cols = get_colors(len(features))
    return((features, cols))


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
        start="1984",
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
            "color": "aliceblue",
            "fillColor": col,
            "fillOpacity": opac}})

    return((feat, ease, lat, lon, stats, details))


def get_null_summary(xrdataset):
    """ """
    percent = {}
    typebool = {"in situ": False, 
                "airborne": False, 
                "spaceborne": False}                             # summarize
    for name, dataset in xrdataset.items():
        t = dataset.attrs["type"]
        dsnull = dataset.isnull().sel(stat="Mean", drop=True)    # get null dataset
        if (not dsnull.data.all()) & (name not in ignore_variables):
            typebool[t] = True
            percent[name] = np.logical_not(dsnull.data).mean() 
    
    return((typebool, percent))


def get_symbology(typebool):
    """ """
    style = {"fill_opacity": 0.4, 
             "color":"white",
             "stroke": True, 
             "weight": 1}
    if typebool["in situ"]:
        style.update({
            "fill_color": "#ff7f00", 
            "fill_opacity": 0.8, 
            "radius": 8})
    if typebool["airborne"]:
        style.update({
            "color": "#377eb8", 
            "stroke": True, 
            "weight": 3,
            "radius": 8})
    return(style)


# ----------------------------------------------------------------------------
# other helpers


def get_ancillary_data(geojson):
    """ """
    features, cols = from_geojson(geojson)            # get features and cols

    layers = []                                       # a temporary list 
    for i, feat in enumerate(features):               # loop over features
        lyrd = get_layer_data(i, feat, opac=0, samp=False)
        lyrm = GeoJSON(data=lyrd[0], hover_style={"color": cols[i]})
        layers.append((i, lyrd[2], lyrd[3], lyrm, lyrd[5]))
    layers = pd.DataFrame(layers, columns=["id","lat","lon","layer","attrs"])

    return(layers)


def fDaymet(xrds, Interval=None, Stack=False):
    """ """
    
    series = [[xrds.tmin, 0, dict(color="darkorange", label="avg min temp")],
              [xrds.tmax, 0, dict(color="darkgreen", label="avg max temp")],
              [xrds.prcp, 1, dict(color="purple", label="avg rate of precip")]]
    
    if (Stack) & (Interval!="year"):
        for i, l in enumerate(series):
            series[i][0] = series[i][0].mean("year")
            series[i][2]["x"] = Interval
    
    fmt0 = [lambda a: a.grid("on", alpha=0.5),
            lambda a: a.set_xlabel(None),
            lambda a: a.legend(loc="upper left"),
            lambda a: a.set_ylabel("degrees Celsius")]
    fmt1 = [lambda a: a.legend(loc="upper right"), 
            lambda a: a.set_ylabel("mm/day")] 

    return((series, fmt0, fmt1))


def fProductivity(xrds, Interval=None, Stack=False):
    """ """
    
    series = [[xrds.GPP_mean, 0, dict(color="darkorange", label="GPP")],
              [xrds.NEE_mean, 0, dict(color="darkgreen", label="NEE")]]
    
    if (Stack) & (Interval!="year"):
        for i, l in enumerate(series):
            series[i][0] = series[i][0].mean("year")
            series[i][2]["x"] = Interval
    
    fmt0 = [lambda a: a.grid("on", alpha=0.5),
            lambda a: a.legend(loc="upper left"),
            lambda a: a.set_xlabel(None),
            lambda a: a.set_ylabel("g m-2 d-1")]
    fmt1 = [lambda a: a.set_title(None)]

    return((series, fmt0, fmt1))


def fSoilMoisture(xrds, Interval=None, Stack=False):
    """ """
    
    fyear = lambda x: x.sel(year=np.unique(x.year)).mean("year")
    
    series = []
    for k,v in {
        "surface": dict(soil_zone="surface"), 
        "rootzone": dict(soil_zone="surface"),
        "in situ": dict(type="in situ"), 
        "airborne": dict(type="airborne"),
        "spaceborne": dict(type="spaceborne")}.items():

        ds = xrds.filter_by_attrs(**v)
        ds = xr.concat(ds.values(), "mean").mean("mean")
        
        p = [ds, 0, dict(label=k)]
        if (Stack)&(Interval!="year"):
            p[0] = fyear(ds)
            p[2]["x"] = Interval
        else:
            p[2]["x"] = "year" if (Stack)&(Interval=="year") else "time"
            
        series.append(p) 
    
    fmt0 = [lambda a: a.set_title(None),
            lambda a: a.set_xlabel(None),
            lambda a: a.grid("on", alpha=0.5),
            lambda a: a.legend(loc="upper left"),
            lambda a: a.set_ylabel("soil moisture volume (m3/m3)")]
    fmt1 = [lambda a: a.set_title(None)]

    return((series, fmt0, fmt1))


#-----------------------------------------------------------------------------


class Plotter:
    """Generates the map/plot side-by-side widget container."""
    
    error = HTML("""<p>Selections can't be plotted <b>""" 
                 """(probably time slider)</b>.</p>""")
    wait = HTML("""<p style="position:relative;top:50%;transform:translateY"""
                """(-50%);">Calculating statistics. Please be patient.</p>""")
    
    dataset_keys = {
        "Soil Moisture": (dict(units="m3/m3"), fSoilMoisture),
        "SMAP GPP/NEE": (dict(units="g m-2 d-1"), fProductivity),
        "Daymet": (dict(source="Daymet"), fDaymet)}
    
    interval_keys = {
        "day": ("d", "dayofyear"), 
        "week": ("w", "week"), 
        "month": ("m", "month"), 
        "year": ("y", "year")}

    
    def __init__(self, layer): 

        self.layer = layer        
        self.xr = layer.xr.sel(stat="Mean", drop=True)
        self.time = pd.to_datetime(self.xr.time.data).strftime("%Y-%m-%d")
        self.samp = layer.samples.samp.tolist()

        # --------------------------------------------------------------------
        # init widgets and set interactivity

        self.dsel = Dropdown(
            options=["Soil Moisture", "SMAP GPP/NEE", "Daymet"],
            value="Soil Moisture",
            layout=Layout(width='auto'))
        self.dsel.observe(self.handler, names="value")
        
        self.intv = Dropdown(
            options=self.interval_keys.keys(), 
            value="day",
            layout=Layout(width='auto'))
        self.intv.observe(self.handler, names="value")
        
        self.stack = ToggleButton(description="Stack", value=False)
        self.stack.observe(self.update_stack, names="value")
        
        # --------------------------------------------------------------------
        # construct ui
        
        self.fig, self.ax0 = plt.subplots(figsize=(6.5, 4))
        self.ax1 = self.ax0.twinx()
        self.fig.tight_layout(pad=2, h_pad=4)
            
        self.mapw = Map(
            layers=(self.layer.layer.layer, layer.points, bmap,), 
            center=(layer.lat, layer.lon), 
            zoom=8, attribution_control=False)
        for p in layer.points.layers:
            p.on_click(self.update_sample)

        self.selections = Box(
            children=[self.dsel, self.intv, self.stack], 
            layout=Layout(
                width="auto",
                display='flex',
                flex_flow='row',
                align_items='stretch'))
        
        self.ui = GridBox(
            children=[VBox([self.selections, self.mapw]), self.fig.canvas], 
            layout=Layout(
                width='100%',
                grid_template_rows="auto",
                grid_template_columns="30% 70%",
                grid_template_areas='''"self.mapw self.fig.canvas"'''))
        
        # --------------------------------------------------------------------        
        # trigger draw
        
        self.handler()

    # ------------------------------------------------------------------------
    
    def update_sample(self, **kwargs):
        """ """
        self.handler()
    
    def update_dataset(self, change):
        """ """
        self.handler()
    
    def update_stack(self, change):
        """ """
        stack = change.new
        self.handler()
        
    def handler(self, change=None):
        """ """
        dataset = self.dsel.value
        interval = self.intv.value
        stack = self.stack.value
        
        # select toggled-on samples
        samp = [s.id for s in self.samp if s.on]
        ds = self.xr.sel(sample=samp)
        
        # filter to dataset-level
        dfilter, dfunc = self.dataset_keys[self.dsel.value]
        ds = ds.filter_by_attrs(**dfilter)
        ds = ds.mean("sample", keep_attrs=True)
        
        # aggregate to interval and 
        intvlstr, stackstr = self.interval_keys[interval]
        if interval!="day":
            ds = ds.resample(time="1"+intvlstr).mean(keep_attrs=True) 
            
        ds.coords["year"] = ds.time.dt.year
        sfunc = lambda x: x.groupby(interval).mean(keep_attrs=True)
        if stack:
            ds.coords[interval] = getattr(ds.time.dt, stackstr)
            ds = ds.groupby("year").apply(sfunc)

        self.plot_config = dfunc(ds, Interval=interval, Stack=stack)
        self.plotter()

    def plotter(self):
        """ """
        self.ax0.clear(); self.ax1.clear()
        
        series, fmt0, fmt1 = self.plot_config
        minx, maxx, count0, count1 = 0,0,0,0
        for i, s in enumerate(series):
            x, xax, xargs = s
            if xax==0: count0 += 1
            else: count1 += 1
            ax = self.ax0 if xax==0 else self.ax1
            try:
                x.plot(ax=ax, **xargs)
            except:
                pass

        for c in fmt0: c(self.ax0)
        for c in fmt1: c(self.ax1)
        if count1==0: self.ax1.axis("off")
        else: self.ax1.axis("on")
        
        self.fig.canvas.draw()



class Sample(object):

    off = {"fill_opacity": 0.4, "stroke": False}

    def __init__(self, i, lat, lon):
        """Inits with id,lat,lon; makes request string, map point."""
        self.id = i
        self.lat = lat 
        self.lon = lon

        self.on = False
        self.pt = CircleMarker(location=(lat, lon), **pt_style)  
        self.dl = smvdownload+"lt={0}&ln={1}&d=smap".format(lat, lon)

    def update(self, **kwargs):
        for arg, val in kwargs.items():
            setattr(self.pt, arg, val)


    def toggle(self, event, type, coordinates):
        style = self.off if self.on else self.symbology   # determine opacity
        self.update(**style)                              # set opacity
        self.on = False if self.on else True              # toggle status


    def submit(self, session):
        """Called by parent. Downloads url. Updates status."""

        self.response = session.get(self.dl)
        self.df = txt_to_pd(self.response.text)            # read to df
        self.xr = get_sample_xr(self)                      # get xr dataset
        self.pt.on_click(self.toggle)                      # callback toggle
        self.summary = get_null_summary(self.xr)           # get null summary
        self.symbology = get_symbology(self.summary[0])
        self.on = True                                     # toggle status on


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
       
            
class JupyterSMV(object):
    """
    App.
    """

    head = HTML("""
    <h3>Batch download Soil Moisture Visualizer datasets for USFS sites:</h3>
    <p>1. Click a polygon on the map.</br>2. Hit submit. </br></br>
    Download locations with an ORANGE dot have in situ observations and those
    with blue stroke have airborne observations. All locations have SMAP
    coverage after ~2016.</p>
    """)
    
    
    def __init__(self, primary=None, anc=None, session=None):

        if session:
            self.session = session
        else:
            raise("You must be logged with Earthdata to use this widget."
                  "Login with the cell above, then run this cell again.")
    
        # -------------------------------------------------------------------
        
        self.polys = LayerGroup()
        self.points = LayerGroup()
        self.apolys = LayerGroup()
        
        self.mapw = Map(
            layers=(bmap, self.apolys, self.polys, self.points,), 
            center=(32.75, -109), 
            zoom=7, 
            width="auto", 
            height="auto",
            scroll_wheel_zoom=True)
        self.submit = Button(**submit_args)
        self.submit.on_click(self.submit_handler)
        self.progress = IntProgress(**progress_args)
        
        # -------------------------------------------------------------------
        
        if primary:                                   # if given, 
            self.load_features(primary)               # load input features
        if anc:
            self.load_ancillary(anc)
        dllayout = [self.head, self.mapw, HBox([self.submit, self.progress])]
        self.downloadui = VBox(dllayout)# + [HBox([self.out1, self.out2])])
        
        # -------------------------------------------------------------------

        self.plotui = Output()
        self.ui = VBox([self.downloadui,  self.plotui])
        display(self.ui)

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
    
    
    def submit_handler(self, b):
        """Resets UI and sends requests to SMV when new submit."""
        self.submit.disabled = True
        
        i = self.selected
        layer_row = self.layers.iloc[i]
        sample = layer_row.samples["samp"].tolist()
        
        self.progress.max = len(sample)            # reset progress bar
        self.progress.min = 0                      
        self.progress.value = 0
        
        for s in sample:                           # loop over sample pts
            s.submit(self.session)                 # download the data
            self.progress.value += 1               # update progress bar
            s.update(**s.symbology)                # update style
        layer_row.layer.dl = True                  # set dl status to True
        
        xrds = xr.concat([s.xr for s in sample], "sample") # make xr dataset
        layer_row.layer.xr = xrds
        self.layers.at[self.selected,"xr"] = xrds 

        with self.plotui:
            p = Plotter(self.layers.iloc[i])
            display(p.ui)
