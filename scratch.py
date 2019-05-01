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


def get_plot(xrds):
    """ """
    
    units = get_options(xrds, "units")
    sources = get_options(xrds, "source")
    
    fig, axs = plt.subplots(
        nrows=len(units), ncols=1, sharex=True, figsize=(20,12))
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