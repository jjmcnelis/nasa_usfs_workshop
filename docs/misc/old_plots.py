rz_mean = rootzone.mean("sample", keep_attrs=True)          # calc mean over sample dimension
rzcat = xr.concat(rz_mean.values(), "cat").mean("cat")      # concat all variable arrays and average

rzcat.coords["year"] = rzcat.time.dt.year                   # add year coordinate
rzcat.coords["month"] = rzcat.time.dt.month                 # add month coordinate               
rzcat.coords["day"] = rzcat.time.dt.dayofyear               # add doy coordinate

rz_10_16 = rzcat.sel(time=slice(str(2010),str(2016)))       # select years 2010-2016
rz_10_16

fig = plt.figure(figsize=(15, 5))                              # init figure
ax = plt.gca()                                                 # get figure axes

for yr in np.unique(rz_10_16.year.data):                       # loop over years
    ln = rz_10_16.sel(time=str(yr)).groupby("month").mean()    # monthly average
    ln.plot.line(x="month", label=str(yr), ax=ax)              # plot

rz_mo = rz_10_16.groupby("month")                              # group by month
mo_mean, mo_std = rz_mo.mean(), rz_mo.std()                    # calc mean, standard deviation
mo_stdlow, mo_stdhi = (mo_mean-mo_std), (mo_mean+mo_std)       # get std bounds

# plot mean, then standard deviation
mo_mean.plot.line(ax=ax, x="month", label="mean", linestyle="--", color="black")
ax.fill_between(mo_mean.month, mo_stdlow, mo_stdhi, color="gray", alpha=0.2)

ax.grid("on", alpha=0.5)                                       # some plot settings
ax.set_ylabel("rootzone m3/m3")
ax.legend(loc=1, bbox_to_anchor=(1.15, 1))
ax.set_title("monthly volumetric soil moisture by year (all sources)") 

# -------------------------------------------------------------------------------------

names = []
for name, dataset in airmoss.items():
    if not dataset.isnull().data.all():
        names.append(name)

fig, axs = plt.subplots(nrows=len(names), ncols=1, sharex=True, figsize=(18, len(names)*4))
fig.tight_layout()
plt.subplots_adjust(hspace=0.000)

for i, name in enumerate(names):
    
    data = airmoss[name]
    ptime = data.time.data
    pmean = data.mean("sample", keep_attrs=True)
    pstd = data.std("sample", keep_attrs=True)
    
    pmean.plot.line(x="time", ax=axs[i], label=data.description, add_legend=False)
    axs[i].fill_between(ptime, (pmean-pstd), (pmean+pstd), color="gray", alpha=0.2)
    axs[i].set_ylabel(name+" "+data.attrs["units"])
    axs[i].grid("on", alpha=0.25)

axs[0].set_title("AirMOSS Soil Moisture Datasets")

# -------------------------------------------------------------------------------------

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(18,8))
fig.tight_layout()
plt.subplots_adjust(hspace=0.000)

for i, depth in enumerate(["rootzone", "surface"]):
    ax = axs[i]
    for platform_type in ["spaceborne", "airborne", "in situ"]:
        
        tds = ds.filter_by_attrs(type=platform_type, soil_zone=depth)
        tdscat = xr.concat(tds.values(), "sample")
        tdsmean = tdscat.resample(time="1w").mean(skipna=True, keep_attrs=True) 
        
        if not all(tdsmean.isnull()):
            tdsmean.plot(ax=ax, label=platform_type)

    ax.legend(loc=2)
    ax.grid("on", alpha=0.25)
    ax.set_ylabel(depth+" (m3/m3)")

axs[0].set_title("volumetric soil moisture by platform type (in situ, airborne, spaceborne)")
axs[0].set_xlabel(None)
plt.show()

# -------------------------------------------------------------------------------------

fig = plt.figure(figsize=(15, 5))                              # init figure
ax = plt.gca()                                                 # get figure axes

for yr in np.unique(rzcat.year.data):                          # loop over years
    ln = rzcat.sel(time=str(yr)).groupby("month").mean()       # monthly average
    ln.plot.line(x="month", label=str(yr), ax=ax)              # plot

rz_mo = rzcat.groupby("month")                                 # group by month
mo_mean, mo_std = rz_mo.mean(), rz_mo.std()                    # calc mean, standard deviation
mo_stdlow, mo_stdhi = (mo_mean-mo_std), (mo_mean+mo_std)       # get std bounds

# plot mean, then standard deviation
mo_mean.plot.line(ax=ax, x="month", label="mean", linestyle="--", color="black")
ax.fill_between(mo_mean.month, mo_stdlow, mo_stdhi, color="gray", alpha=0.2)

# also get SMAP GPP mean by month       
gpp = ds_2012_2019["GPP_mean"].mean("sample", keep_attrs=True) # calc mean over sample dim 
gpp_mo = gpp.groupby("month")                                  # group by day
gpp_mean, gpp_std = gpp_mo.mean(), gpp_mo.std()                # get mean, std
gpp_stdlow, gpp_stdhi = (gpp_mean-gpp_std), (gpp_mean+gpp_std) # get std bounds

# add another y axis to the second plot and plot mean, then standard deviation
ax2 = ax.twinx()  
gpp_mean.plot.line(ax=ax2, x="month", label="GPP (SMAP)", color="midnightblue")
ax2.fill_between(gpp_mean.month, gpp_stdlow, gpp_stdhi, color="midnightblue", alpha=0.2)

ax.grid("on", alpha=0.5)                                       # some plot settings
ax.set_ylabel("rootzone m3/m3")
ax.legend(loc=1, bbox_to_anchor=(1.15, 1))
ax.set_title("monthly volumetric soil moisture by year (all sources)") 