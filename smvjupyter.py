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
from ipyleaflet import Map, LayerGroup, GeoJSON, CircleMarker, WidgetControl
from ipywidgets import RadioButtons, Checkbox, Box, Accordion, IntSlider, Text, Dropdown, ToggleButton
from ipywidgets import Layout, Button, IntProgress, Output, HBox, VBox, HTML, interactive, Image


disabled_sources =  ["FLUXNET", "MODIS", "GRACE"]
ignore_variables = ["sample","time","stat","lat","lon"] # "PBOH2O"

# ----------------------------------------------------------------------------
# app settings

#font = {"family": "normal", "weight": "normal", "size": 16}  # matplotlib ->
#plt.rcParams['figure.figsize'] = [14, 5]
#plt.rc("font", **font) # <- some matplotlib settings

warnings.filterwarnings('ignore')
auth = dict(ORNL_DAAC_USER_NUM=str(32863))

smvdownload = "https://daac.ornl.gov/cgi-bin/viz/download.pl?"
smvdatasets = pd.read_csv(
    "docs/smvdatasets.csv", 
    index_col="dataset", 
    header=0)

obslegend = open("docs/img/obslegend2.png", "rb")
obslegendimg = obslegend.read()
obslegendwg = Image(value=obslegendimg, format='png', width=125, height=75)

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


def poly_mapper(lyr, mw="25%", ow="75%", zoom=8):
    """Generates the map/plot side-by-side widget container."""

    l = (lyr["layer"].layer, lyr.points, bmap,)
    m = Map(layers=l, center=(lyr["lat"], lyr["lon"]), zoom=zoom, width=mw)
    o = Output(layout={"width": ow})
    
    return((m,o))


# ----------------------------------------------------------------------------

def get_colors(n, cmap=cm.Set3):
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


# ease data
latf = "docs/EASE2_M09km.lats.3856x1624x1.double"
lonf = "docs/EASE2_M09km.lons.3856x1624x1.double"

lats = np.fromfile(latf, dtype=np.float64).flatten() 
lons = np.fromfile(lonf, dtype=np.float64).flatten()
crds = np.dstack((lats,lons))[0]


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
    style = {#"fill_color": "#cbd5e8", 
             "fill_opacity": 0.4, 
             "color":"white",
             "stroke": True, 
             "weight": 1}
    if typebool["in situ"]:
        style.update({
            "fill_color": "#ff7f00", 
            "fill_opacity": 0.8, 
            "radius": 8})
    if typebool["airborne"]:
        style.update({"color": "#377eb8", "stroke": True, "radius": 8})
    return(style)


# ----------------------------------------------------------------------------
# other helpers

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


def get_nan_summary(xrdataset): 
    """ """
    nandict = {"in situ": {}, "airborne": {}, "spaceborne": {}}

    for pt in nandict.keys():

        # for percentage: ----------------------------------------------------
        """# get the datasets for the current platform
        ds = xrds.filter_by_attrs(type=pt).sel(stat="Mean", drop=True)

        # get fraction of null dataset by timestep for all samples
        dsnull = ds.isnull()

        # get variables with nodata; variables with data; valid counts
        nodata, yesdata, data = [], [], {}
        for name, dataset in dsnull.items():
            if dataset.data.all(): #dataset.isnull().mean()==1
                nodata.append(name)
            else:
                yesdata.append(name)
                dataset.data = np.logical_not(dataset.data)
                data[name] = dataset.mean("sample").data"""
        # --------------------------------------------------------------------

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


def get_output_layout(w="95%", b="1px solid lightgray"):
    """ """
    return({"width": w, "border": b})


def get_options(xrds, attribute):
    """ """
    
    options = []
    for name, xrda in xrds.items():
        isnull = xrda.isnull().data.all()
        isignore = name in ignore_variables
        if not any([isnull, isignore]):
            option = name if attribute=="dataset" else xrda.attrs[attribute]
            options.append(option)
            
    return(sorted(list(set(options))))
    

def get_checkboxes(xrds, group):
    """ """

    checkboxes = [Checkbox(
        description=str(o), 
        value=True, 
        indent=False,
        layout=Layout(width='auto')
    ) for o in get_options(xrds, group)]
    
    return(checkboxes)


def get_plot(xrds):
    """ """
    
    units = get_options(xrds, "units")
    sources = get_options(xrds, "source")
    
    fig, axs = plt.subplots(nrows=len(units), ncols=1, sharex=True, figsize=(20,12))
    plt.subplots_adjust(hspace=0.000)
    
    for i, u in enumerate(units):
        x = xrds.filter_by_attrs(units=u)
        ax = axs[i] if len(units)>1 else axs
        for n, d in x.items():
            d.mean("sample").plot.line(
                x="time", 
                label=n, 
                ax=ax, 
                add_legend=False)
        ax.set_xlabel(None)
        ax.set_ylabel(str(u))
        ax.set_title(None)
        ax.legend(loc=0, framealpha=1)

    plt.show()
            

#-----------------------------------------------------------------------------

class Sample(object):

    off = {"fill_color": "#cbd5e8", "fill_opacity": 0.1, "stroke": False}

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


    def submit(self, session=None):
        """Called by parent. Downloads url. Updates status."""

        if session:
            self.response = session.get(self.dl)
        else:
            self.response = requests.get(self.dl, cookies=auth)

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
    
    def getui(self):
        """ """
        self.nan = get_nan_summary(self.xr)
        #self.opt = get_options(self.xr)

def get_options(xrds, attribute):
    """ """
    
    options = []
    for name, xrda in xrds.items():
        isnull = xrda.isnull().data.all()
        isignore = name in ignore_variables
        if not any([isnull, isignore]):
            option = name if attribute=="dataset" else xrda.attrs[attribute]
            options.append(option)
            
    return(sorted(list(set(options))))
    

def get_checkboxes(xrds, group):
    """ """

    checkboxes = [Checkbox(
        description=str(o), 
        value=True, 
        indent=False,
        layout=Layout(width='auto')
    ) for o in get_options(xrds, group)]
    
    return(checkboxes)


def get_plot(xrds):
    """ """
    
    units = get_options(xrds, "units")
    sources = get_options(xrds, "source")
    
    fig, axs = plt.subplots(nrows=len(units), ncols=1, sharex=True, figsize=(20,12))
    plt.subplots_adjust(hspace=0.000)
    
    for i, u in enumerate(units):
        x = xrds.filter_by_attrs(units=u)
        ax = axs[i] if len(units)>1 else axs
        for n, d in x.items():
            d.mean("sample").plot.line(
                x="time", 
                label=n, 
                ax=ax, 
                add_legend=False)
        ax.set_xlabel(None)
        ax.set_ylabel(str(u))
        ax.set_title(None)
        ax.legend(loc=0, framealpha=1)

    plt.show()
            

class Plotter:
    """Generates the map/plot side-by-side widget container."""
    
    error = HTML("""<p>Please select one or more platform types.</p>""")
    
    groups = ["dataset", "source"]
    intervals = {"day": "d", "week": "w", "month": "m", "year": "y"}
    
    def __init__(self, layer, debug=False): 

        self.layer = layer
        self.debug = debug
        
        self.xr = layer.xr.sel(stat="Mean", drop=True)
        self.filtxr = layer.xr

        # selection container
        self.selected = dict(
            interval="day",
            stack=False,
            group="dataset",
            groupoptions={})
        
        # --------------------------------------------------------------------
        # selection ui
        
        # platform type options
        self.selected["type"] = get_options(self.xr, "type")
        self.platformtypes = Box([], layout=Layout(
            width="auto",
            display='flex',
            flex_flow='column',
            align_items='stretch'))
        checkboxes = []
        for t in self.selected["type"]:
            chkbx = Checkbox(
                value=True, 
                description=t, 
                indent=False, layout=Layout(width='auto'))
            chkbx.observe(self.type_handler, names="value")
            checkboxes.append(chkbx)
        self.platformtypes.children = checkboxes
        
        # interval options
        self.interval = RadioButtons(
            options=self.intervals.keys(), 
            value="day", 
            layout=Layout(width='auto'))
        self.interval.observe(self.options_handler, names="value")
        
        # dataset options      
        self.chkbxoutput = Box(
            children=[], 
            layout=Layout(
                height="200px",
                width="auto",
                display='flex',
                flex_flow='column',
                align_items='stretch',
                overflow_y="scroll",
                border="1px solid gray"))
        self.group_handler()

        # minimap ui
        #self.mapw = Map(                                             # map widget
        #    layers=(self.layero.layer, layer.points, bmap,), 
        #    center=(layer.lat, layer.lon), 
        #    zoom=zoom, width=mw)
        #self.output = Output(layout={"width": ow})                   # Output widget 
        #self.mapplotui = HBox([self.mapw, self.output])              # Map/plot ui        
        
        # selection ui (left panel) ---------------------- COMBINED
        self.selectui = Box([
            HTML("<b>Platform: </b>"),
            self.platformtypes,
            HTML("<b>Interval: </b>"),
            self.interval,
            HTML("<b>Datasets: </b>"),
            self.chkbxoutput
        ], layout=Layout(
            width="20%",
            display='flex',
            flex_flow='column',
            align_items='stretch'))

        # --------------------------------------------------------------------
        # ui
        
        self.output = Output(layout={"width": "auto"})
        self.ui = HBox([self.selectui, self.output])
        self.to_output()
        
    # ------------------------------------------------------------------------

    ##### pppppppppppppppppppppppppppppppppppppppppp fixes needed
    def type_handler(self, *args, **kwargs):
        """ """
        types = []
        for t in self.platformtypes.children:
            if t.value:
                types.append(t.description)
        print(types)
        self.selected["type"] = types
        self.group_handler()
        self.to_output()
    
    
    def group_handler(self, *args, **kwargs):
        """ """
        
        types = self.selected["type"]
        filtxr = self.xr.filter_by_attrs(type=lambda t: t in types)
        self.checkboxes = get_checkboxes(filtxr, "dataset")
        for c in self.checkboxes:
            c.observe(self.group_chk_handler, names="value")
        self.chkbxoutput.children = self.checkboxes
        
        
    def group_chk_handler(self, *args, **kwargs):
        """ """
        selections = []
        for c in self.checkboxes:
            if c.value:
                selections.append(c.description)
        self.selected["groupoptions"] = selections
        self.to_output()
    ###      ---------------------------------------------- fixes needed    
     
     
        
    def options_handler(self, *args, **kwargs):
        """ """
        for n, w in {"interval": self.interval}.items():
            self.selected[n] = w.value
        self.to_output()
        
        
    def to_output(self, *args, **kwargs):
        """ """
        self.output.clear_output()
        
        types = self.selected["type"]
        
        interval = self.interval.value
        filtxr = self.xr.filter_by_attrs(type=lambda t: t in types)
        
        checked = self.selected["groupoptions"]
        filtxr = filtxr[checked]

        with self.output:
            
            units = get_options(filtxr, "units")
            sources = get_options(filtxr, "source")

            fig, axs = plt.subplots(
                nrows=len(units), 
                ncols=1, 
                sharex=True, 
                figsize=(15,10))
            plt.subplots_adjust(hspace=0.000)

            for i, u in enumerate(units):
                ax = axs[i] if len(units)>1 else axs
                x = filtxr.filter_by_attrs(units=u)
                
                if interval!="day":
                    iv = "1"+self.intervals[interval]
                    x = x.resample(time=iv).mean()

                for n, d in x.items():
                    d.mean("sample").plot.line(label=n, ax=ax, add_legend=False)

                ax.set_xlabel(None)
                ax.set_ylabel(str(u))
                ax.set_title(None)
                ax.legend(loc=0, framealpha=1)

            plt.show()
            
            
class JupyterSMV(object):
    """
    App.
    """

    head = HTML("""
    <h3>Soil Moisture Visualizer</h3>
    <p>Do such and such.</br>1. </br>a. </br>2. </br>b.</p>
    """)
    
    
    def __init__(self, primary=None, anc=None, session=None):

        self.session = session

        self.polys = LayerGroup()
        self.points = LayerGroup()
        self.apolys = LayerGroup()
        
        # -------------------------------------------------------------------
        
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
        dllayout = [self.mapw, HBox([self.submit, self.progress])]
        self.downloadui = VBox(dllayout)# + [HBox([self.out1, self.out2])])
        
        # -------------------------------------------------------------------
        
        self.plotui = Output()
        self.body = Accordion(
            children=[self.downloadui, self.plotui], 
            layout=Layout(width="auto", height="80%"))
        self.body.set_title(0, '1. Download')
        self.body.set_title(1, '2. Plot')
        self.ui = VBox([self.head,  self.body])

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
            s.submit(session=self.session)         # download the data
            self.progress.value += 1               # update progress bar
            s.update(**s.symbology)                # update style
        layer_row.layer.dl = True                  # set dl status to True
        
        xrds = xr.concat([s.xr for s in sample], "sample") # make xr dataset
        layer_row.layer.xr = xrds
        self.layers.at[self.selected,"xr"] = xrds 
        self.body.selected_index = 1
        
        layer_row.layer.getui()
        self.plotter = Plotter(layer_row.layer)
        self.plotui.clear_output()
        with self.plotui:
            display(self.plotter.ui)
