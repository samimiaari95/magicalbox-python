import pandas as pd
import numpy as np
import netCDF4 as nc
import SLOTH.sloth.toolBox
import SLOTH.sloth.IO
import SLOTH.sloth.PlotLib
import SLOTH.sloth.mapper
import os
import re
import shutil
from datetime import date, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import LineString



def model_outputs(modelname):
    # get the original sami's folder path "miaari1"
    proj_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    folders = ["2020010100", "2020020100", "2020030100", "2020040100", "2020050100", "2020060100", "2020070100", "2020080100", "2020090100", "2020100100", "2020110100"]
    all_files = []
    for folder in folders:
        output_dir = os.path.join(proj_path, "FZJ-IBG3_WorkflowStarter", "simres", folder, f"{modelname}")
        files = [os.path.join(output_dir, x) for x in os.listdir(output_dir) if x.endswith(".nc") and "h1" not in x]# and "p" not in x]
        all_files = all_files + files

    return all_files

def prepare_3D_4D_dfs(files):
    file4D = [x for x in files if "p" in os.path.basename(x)]
    file3D = [x for x in files if "p" not in os.path.basename(x)]
    ncfile = nc.Dataset(file3D[0])
    variables = ncfile.variables
    cols = [f"{ncfile[var].long_name} ({ncfile[var].units})" for var in variables if len(ncfile[var].dimensions) == 3]
    columns = ["year", "month", "day", "time"] + cols
    df3D = pd.DataFrame(columns=columns)

    #ncfile = nc.Dataset(file4D[0])
    #variables = ncfile.variables
    #cols = [f"{ncfile[var].long_name} ({ncfile[var].units})" for var in variables if len(ncfile[var].dimensions) == 4]
    #columns = ["year", "month", "day", "time"] + cols
    #df4D = pd.DataFrame(columns=columns)
    #pressure_values = list(ncfile["pressure"][:])
    #pressure_unit = ncfile["pressure"].units
    #dict4d = {x:df4D for x in pressure_values}
    dict4d = {}
    return df3D, dict4d
    

def nc_to_csv(files):
    df3D, dict4D = prepare_3D_4D_dfs(files)
    for file in files:
        
        filename = os.path.basename(file)
        #if not filename=="lffd2020011203p.nc":
        #    print(filename)
        #    continue
        print(f"working on {os.path.basename(file)}")
        date = re.findall("[0-9]+", filename)[0]
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        time = date[8:]
        ncfile = nc.Dataset(file)
        variables = ncfile.variables
        df3 = pd.DataFrame(columns=df3D.columns)
        df3["year"] = [year]
        df3["month"] = [month]
        df3["day"] = [day]
        df3["time"] = [time]
        #for var in variables:
        #    dim = len(ncfile[var].dimensions)
        #    if dim==4:
        #        print(ncfile[var].long_name)
        #return
        for var in variables:
            dim = len(ncfile[var].dimensions)
            if dim <= 2:
                continue
            #if not (dim == 4 and var == "T"):
            #    continue
            update_3D = False
            varname = ncfile[var].long_name
            varunit = ncfile[var].units
            value = ncfile[var][:]

            if dim == 3:
                value = np.mean(value[0])
                df3[f"{varname} ({varunit})"] = [value]
                update_3D = True

            elif dim == 4:
                continue
                for i, val in enumerate(dict4D.keys()):
                    mean_value = np.mean(value[0][i])
                    #if not (year=="2020" and month=="01" and day=="12" and val==100000.0):
                    #    continue
                    df = dict4D[val]
                    #print(val)
                    #print(mean_value)
                    # NOTE probably the problem is somewhere here in the if conditions
                    #if df.empty:
                    #    print("df empty")
                    #    if df.loc[(df["year"]==year) & (df["month"]==month) & (df["day"]==day) & (df["time"]==time)].empty:
                    #        print("it's working")
                    if df.empty or df.loc[(df["year"]==year) & (df["month"]==month) & (df["day"]==day) & (df["time"]==time)].empty:
                        df = pd.DataFrame(columns=dict4D[val].columns)
                        df["year"] = [year]
                        df["month"] = [month]
                        df["day"] = [day]
                        df["time"] = [time]
                        df[f"{varname} ({varunit})"] = [mean_value]
                    #    print(mean_value)
                    #    print(f"{varname} ({varunit})")
                        dict4D[val] = pd.concat([dict4D[val], df], ignore_index=True)
                        dict4D[val].reset_index(inplace=True, drop=True)
                    else:
                    #    print("we went to the else")
                        df.loc[(df["year"]==year) & (df["month"]==month) & (df["day"]==day) & (df["time"]==time), f"{varname} ({varunit})"] = mean_value
                    #    print(df.loc[(df["year"]==year) & (df["month"]==month) & (df["day"]==day) & (df["time"]==time), f"{varname} ({varunit})"])
                        dict4D[val] = df
                    #    print(mean_value)
                    #    print(f"{varname} ({varunit})")
        if update_3D:
            df3D = pd.concat([df3D, df3], ignore_index=True)
            df3D.reset_index(inplace=True, drop=True)
    #year = "2020"
    #month = "01"
    #day = "12"
    #time = "03"
    #for key in dict4D.keys():
    #    print(key)
    #    print(dict4D[key].loc[(dict4D[key]["year"]==year) & (dict4D[key]["month"]==month) & (dict4D[key]["day"]==day) & (dict4D[key]["time"]==time), "temperature (K)"])
        
    return df3D, dict4D


def sort_cols(filepath):
    df = pd.read_csv(filepath)
    df.sort_values(by=["year", "month", "day", "time"], inplace=True)
    df.drop_duplicates(subset=["year", "month", "day", "time"], inplace=True, keep="last")
    df.reset_index(inplace=True, drop=True)
    df.to_csv(filepath, index=False)


def main():
    modelname = "cosmo"
    files = model_outputs(modelname)
    df3D, dict4D = nc_to_csv(files)
    proj_path = os.path.dirname(os.path.realpath(__file__))
    df3D.to_csv(os.path.join(proj_path, "cosmo_3Dvariables.csv"), index=False)

    #for df_name, df in dict4D.items():
    #    df.to_csv(os.path.join(proj_path, f"{df_name} Pa.csv"), index=False)

    print("sorting...")
    csv_files = [os.path.join(proj_path, x) for x in os.listdir(proj_path) if x.endswith(".csv")]
    for csv in csv_files:
        print(csv)
        sort_cols(csv)


def CLM_outputs():
    modelname = "clm"
    files = model_outputs(modelname)
    df = pd.DataFrame(columns=["year", "month", "day", "time", "qflx_evap_soi + qflx_evap_veg + qflx_tran_veg (mm H2O/s)"])
    for file in files:
        filename = os.path.basename(file)
        print(f"working on {os.path.basename(file)}")
        date = re.findall("[0-9]{2,4}-[0-9]{1,2}-[0-9]{1,2}", filename)[0]
        print(date)
        ncfile = nc.Dataset(file)
        # get the produnce region mask
        lat2D = ncfile["latixy"][:]
        lon2D = ncfile["longxy"][:]
        prudName = 'ME'
        prudenceMask = my_get_prudenceMask(lat2D, lon2D, prudName)
        #variables = ncfile.variables
        var = "QFLX_EVAP_TOT"
        timesteps = ["00", "03", "06", "09", "12", "15", "18", "21"]
        for i, t_step in enumerate(timesteps):
            value = ncfile[var][i]
            masked_values = np.ma.masked_where(prudenceMask==False, value)
            mean_val = np.mean(masked_values)

            df_temp = pd.DataFrame(columns=df.columns)
            #df_temp["date"] = [f"{date}-{t_step}"]
            df_temp["year"] = [date.split("-")[0]]
            df_temp["month"] = [date.split("-")[1]]
            df_temp["day"] = [date.split("-")[2]]
            df_temp["time"] = [t_step]
            df_temp["qflx_evap_soi + qflx_evap_veg + qflx_tran_veg (mm H2O/s)"] = [np.mean(mean_val)]
            df = pd.concat([df, df_temp], ignore_index=True)
            df.reset_index(inplace=True, drop=True)
        print(df)
    prjpath = os.path.dirname(os.path.realpath(__file__))
    df.to_csv(os.path.join(prjpath, "CLM_totevap.csv"), index=False)
    return

def modify_csv():
    prjpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs")
    files = [os.path.join(prjpath, x) for x in os.listdir(prjpath) if x.endswith(".csv")]
    for file in files:
        df = pd.read_csv(file)
        S_date = pd.Series()
        for i in range(len(df)):
            date = f"{df['month'].iloc[i]}-{df['day'].iloc[i]}-{df['time'].iloc[i]}"
            s = pd.Series(date)
            S_date = S_date.append(s, ignore_index=True)
            S_date = S_date.reset_index(drop=True)
        df["date"] = S_date
        print(df)
        df.to_csv(file, index=False)


def my_get_prudenceMask(lat2D, lon2D, prudName):
    """ return a prudance mask

    Return a boolean mask-array (True = masked, False = not masked) based on
    a passed set of longitude and latitude values and the name of the prudence
    region.
    The shape of the mask-array is set equal to the shape of input lat2D.
    Source: http://prudence.dmi.dk/public/publications/PSICC/Christensen&Christensen.pdf p.38

    Input values:
    -------------
    lat2D:    ndarray
        2D latitude information for each pixel
    lon2D:    ndarray
        2D longitude information for each pixel
    prudName: str
        Short name of prudence region

    Return value:
    -------------
    prudMask: ndarray
        Ndarray of dtype boolean of the same shape as lat2D.
        True = masked; False = not masked
    """
    if (prudName=='BI'):
        prudMask = np.where((lat2D < 50.0) | (lat2D > 59.0)  | (lon2D < -10.0) | (lon2D >  2.0), False, True)
    elif (prudName=='IP'):
        prudMask = np.where((lat2D < 36.0) | (lat2D > 44.0)  | (lon2D < -10.0) | (lon2D >  3.0), True, False)
    elif (prudName=='FR'):
        prudMask = np.where((lat2D < 44.0) | (lat2D > 50.0)  | (lon2D < -5.0) | (lon2D >  5.0), True, False)
    elif (prudName=='ME'):
        #prudMask = np.where((lat2D < 48.0) | (lat2D > 55.0)  | (lon2D < 2.0) | (lon2D >  16.0), False, True)
        prudMask = np.where((lat2D < 48.0) | (lat2D > 55.0)  | (lon2D < 2.0) | (lon2D >  16.0), False, True)
    elif (prudName=='SC'):
        prudMask = np.where((lat2D < 55.0) | (lat2D > 70.0)  | (lon2D < 5.0) | (lon2D >  30.0), True, False)
    elif (prudName=='AL'):
        prudMask = np.where((lat2D < 44.0) | (lat2D > 48.0)  | (lon2D < 5.0) | (lon2D >  15.0), True, False)
    elif (prudName=='MD'):
        prudMask = np.where((lat2D < 36.0) | (lat2D > 44.0)  | (lon2D < 3.0) | (lon2D >  25.0), False, True)
    elif (prudName=='EA'):
        prudMask = np.where((lat2D < 44.0) | (lat2D > 55.0)  | (lon2D < 16.0) | (lon2D >  30.0), True, False)
    else:
        print(f'prudance region {prudName} not found --> EXIT')
    countmask = 0
    countnotmasked = 0
    for x in prudMask:
        for y in x:
            if y == True:
                countmask += 1
            elif y == False:
                countnotmasked += 1
    print(countmask)
    print(countnotmasked)
    return prudMask


def prudence_mask(mask, ncfile=None):
    if not ncfile:
        filename = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/postpro/2020_01/T_2M_ts.nc"
        ncfile = nc.Dataset(filename)

    lat2D = ncfile["lat"][:]
    lon2D = ncfile["lon"][:]
    print(lat2D.shape)
    print(lon2D.shape)
    prudName = 'ME'
    #prudenceMask = my_get_prudenceMask(lat2D, lon2D, prudName)
    print(f"prudance mask shape: {mask.shape}")
    evap_var = ncfile["T_2M"][:]
    print(f"temperature shape: {evap_var.shape}")
    timeseries = []
    #for timestep in evap_var:

        #masked_values = np.ma.masked_where(prudenceMask==False, timestep)
    #    masked_values = np.ma.masked_where(mask==0, timestep)
    #    meanvalue = np.mean(masked_values)
    #    timeseries.append(meanvalue)
    
    #for i, x in enumerate(mask[0]):
    #    for j, y in enumerate(x):
    #        if i==227 and j==175:
    #            mask[0][i][j] = 1
    #            print("only one point")
    #        else:
    #            mask[0][i][j] = 0
    masked_example = np.ma.masked_where(mask==0, evap_var[0])
    return masked_example

def open_nc():
    filepath = "/p/scratch/cslts/miaari1/TSMP_WorkflowStarter/simres/MainRun/1970010100/clm/clmoas.clm2.h0.1970-01-01-00900.nc"
    ncfile = nc.Dataset(filepath)
    variables = ncfile.variables
    for var in variables:
        print(ncfile[var])

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def load_npy():
    region = 'SC'
    #filename = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/src/SLOTH/data/example_ClimateMeans/intervalTime_day_TOT_PREC_masked.npy"
    filename = f"/p/scratch/cslts/miaari1/python_test/SLOTH/data/example_ClimateMeans/{region}/intervalMean_day_TOT_PREC_{region}.npy"
    arr = np.load(filename)
    #print(arr.shape)
    #total = np.nanmean(arr)
    #print(total)
    #year = total*365*8
    #print(year)
    #return
    daily_avg = []
    for day in arr:
        print(np.nanmean(day))
        daily_avg.append(np.nanmean(day))
        #daily_values = []
        #for x in day:
            #print(len(x))
        #    x = x[~np.isnan(x)]
        #    if len(x) != 0:
        #        daily_values.extend(x)
        #if daily_values:
        #    daily_avg.append(np.mean(daily_values))
    
    dates = []
    months = []
    days = []
    start_date = date(2020, 1, 1)
    end_date = date(2020, 12, 1)
    for single_date in daterange(start_date, end_date):
        onedate = single_date.strftime("%Y-%m-%d")
        if onedate != "2020-02-29":
            dates.append(onedate)
            months.append(onedate.split("-")[1])
            days.append(onedate.split("-")[-1])


    #m = np.ndarray.mean(d)
    # TODO save it to csv
    dic = {"month": months, "day": days, "Precipitation (mm)": daily_avg}
    df = pd.DataFrame(dic)
    df.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Precipitation_day_{region}.csv"), index=False)


def my_dynamic_get_prudenceMask(lat2D, lon2D, prudName):

    if (prudName=='BI'):
        prudMask = np.where((lat2D < 50.0) | (lat2D > 59.0)  | (lon2D < -10.0) | (lon2D >  2.0), False, True)
    elif (prudName=='IP'):
        prudMask = np.where((lat2D < 36.0) | (lat2D > 44.0)  | (lon2D < -10.0) | (lon2D >  3.0), False, True)
    elif (prudName=='FR'):
        prudMask = np.where((lat2D < 44.0) | (lat2D > 50.0)  | (lon2D < -5.0) | (lon2D >  5.0), False, True)
    elif (prudName=='ME'):
        #prudMask = np.where((lat2D < 48.0) | (lat2D > 55.0)  | (lon2D < 2.0) | (lon2D >  16.0), False, True)
        prudMask = np.where((lat2D < 48.0) | (lat2D > 55.0)  | (lon2D < 2.0) | (lon2D >  16.0), False, True)
    elif (prudName=='SC'):
        prudMask = np.where((lat2D < 55.0) | (lat2D > 70.0)  | (lon2D < 5.0) | (lon2D >  30.0), False, True)
    elif (prudName=='AL'):
        prudMask = np.where((lat2D < 44.0) | (lat2D > 48.0)  | (lon2D < 5.0) | (lon2D >  15.0), False, True)
    elif (prudName=='MD'):
        prudMask = np.where((lat2D < 36.0) | (lat2D > 44.0)  | (lon2D < 3.0) | (lon2D >  25.0), False, True)
    elif (prudName=='EA'):
        prudMask = np.where((lat2D < 44.0) | (lat2D > 55.0)  | (lon2D < 16.0) | (lon2D >  30.0), False, True)
    else:
        print(f'prudance region {prudName} not found --> EXIT')
    return prudMask


def dynamic_prudence_mask(lat2D, lon2D, vardata):
    prudName = 'ME'
    prudenceMask = my_get_prudenceMask(lat2D, lon2D, prudName)
    evap_var = vardata
    for i, timestep in enumerate(evap_var):

        masked_values = np.ma.masked_where(prudenceMask==False, timestep)
        #meanvalue = np.mean(masked_values)
        evap_var[i] = masked_values
    
    return evap_var

def watershed_mask(mask, data):

    for i, timestep in enumerate(data):
        masked_data = np.ma.masked_where(mask==0, timestep)
        data[i] = masked_data
    
    return data

def basin_precip_2020():
    watershed_filename = "/p/scratch/cslts/miaari1/python_test/rhineriverbasin_slopes.npy"
    riverbasin = np.load(watershed_filename)
    all_year = {'day': [], 'month': [], 'Precipitation (mm)': []}
    for i in range(1,12):
        i = f"{i}"
        i = i.zfill(2)
        print(i)
        filepath = f"/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/postpro/2020_{i}/TOT_PREC_ts.nc"
        with nc.Dataset(filepath, 'r') as nc_file:
            data  = nc_file.variables["TOT_PREC"]
            data         = data[...]
            #lon2D        = nc_file["lon"][:]
            #lat2D        = nc_file["lat"][:]
            data         = watershed_mask(riverbasin, data)
        monthly_prec = []
        for j in range(0, len(data)):
            timestep_total = np.nanmean(data[j]) #*3
            monthly_prec.append(timestep_total)
        day = 0
        for k in range(0, len(monthly_prec), 8):
            day += 1
            all_year['day'].append(day)
            all_year['month'].append(i)
            daily_prec = np.sum(monthly_prec[k:k+8])
            all_year['Precipitation (mm)'].append(daily_prec)
    
    df = pd.DataFrame(all_year)
    df.to_csv("./rhineriver_slopes_precip.csv", index=False)
    return

def initialize_mapper():
    slopexfile = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/geo/TSMP_EU11/static/parflow/slopex.sa"
    slopeyfile = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/geo/TSMP_EU11/static/parflow/slopey.sa"
    slopex = SLOTH.sloth.IO.readSa(slopexfile)[0]
    slopey = SLOTH.sloth.IO.readSa(slopeyfile)[0]
    all_lon, all_lat = get_lon_lat()
    all_lon = all_lon.filled(0)
    all_lat = all_lat.filled(0)
    print(type(all_lon))

    Mapper  = SLOTH.sloth.mapper.mapper(SimLons=all_lon, SimLats=all_lat,
                        ObsLons=np.array([6.96329]), ObsLats=np.array([50.93696]),
                        ObsIDs=np.array([1]), ObsMeanArea=np.array([144.232]))
    Mapper.MapBestCatchment(search_rad=3, dy=12500, dx=12500, slopex=slopex, slopey=slopey)


def get_lonlat_px_index(point_lon, point_lat):
    # get index of a point that has these coordinates
    filepath = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/postpro/2020_01/T_2M_ts.nc"
    ncfile = nc.Dataset(filepath)
    lon = ncfile["lon"]
    lat = ncfile["lat"]
    print(f"lon shape: {lon.shape}")
    print(f"lat shape: {lat.shape}")
    # not sure because i get the rotated ones
    min_lon = 100
    min_lat = 100
    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            diff_lon = abs(lon[i][j]-point_lon)
            diff_lat = abs(lat[i][j]-point_lat)
            if diff_lon < min_lon and diff_lat < 0.11:# and diff_lat < min_lat:
            #if diff_lon < min_lon and diff_lat < min_lat:
                min_lon = diff_lon
                #min_lat = diff_lat
                #my_lat = i
                my_lon = j
            if diff_lat < min_lat and diff_lon < 0.11:
                #min_lon = diff_lon
                min_lat = diff_lat
                my_lat = i
               # my_lon = j

    print(f"lat pixel index: {my_lat}")
    print(f"lon pixel index: {my_lon}")
    print(f"lon diff: {min_lon}")
    print(f"lat diff: {min_lat}")
    print(f"lat: {lat[my_lat][my_lon]}")
    print(f"lon: {lon[my_lat][my_lon]}")
    return my_lon, my_lat
    for i, y in enumerate(lon):
        for j, x in enumerate(y):
            diff_lon = abs(x-point_lon)
            if diff_lon < min_lon:
                min_lon = diff_lon
                my_lon = j
                save_i = i

    for i, y in enumerate(lat):
        diff_lat = abs(y[my_lon]-point_lat)
        if diff_lat < min_lat:
            min_lat = diff_lat
            my_lat = i

    #for i, y in enumerate(lon):
    #    diff_lon = abs(x-point_lon)
    #        if diff_lon < min_lon:

    #for i, y in enumerate(lat):
    #    for j, x in enumerate(y):
    #        diff_lat = abs(x-point_lat)
    #        if diff_lat < min_lat:
    #            min_lat = diff_lat
    #            my_lat = i

    print(f"min lon diff: {min_lon}")
    print(save_i, my_lon)
    print(f"the best lon provide lat of: {lat[save_i][my_lon]}")
    print(f"min lat diff: {min_lat}")
    return my_lon, my_lat

def watershed_from_slopes(slopex, slopey, px_x, px_y):
    #slopexfile = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/geo/TSMP_EU11/static/parflow/slopex.sa"
    #slopeyfile = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/geo/TSMP_EU11/static/parflow/slopey.sa"
    #slopex = SLOTH.sloth.IO.readSa(slopexfile)
    #slopey = SLOTH.sloth.IO.readSa(slopeyfile)
    catchment = SLOTH.sloth.toolBox.calc_catchment(slopex, slopey, px_x, px_y)
    #plt.imshow(catchment, origin="lower", interpolation="none")
    #plt.savefig("./catchment.png")
    counttrue = 0
    countfalse = 0
    for x in catchment:
        for y in x:
            if y == 1:
                counttrue += 1
            elif y == 0:
                countfalse += 1
    print(f"Catchment area number of pixels: {counttrue}")
    print(f"Catchment corresponding x {px_x} and y {px_y}")
    return catchment, counttrue, px_x, px_y

def test_plot(org_data, data):
    filename = "/p/scratch/cslts/miaari1/python_test/SLOTH/data/example_ClimateMeans/MD/climate_day_TOT_PREC_masked.npy"
    if len(data)==0:
        data = np.load(filename)
    # plot if meanInterval='month'
    #if meanInterval == 'month':
    title = "testing mask plot"
    varName = "temperature"
    kwargs = {
            'title': f'{title}',
            #'title': '\n'.join(tmp_titlesubstr),
            'infostr': False,
            #'var_vmax': 1,
            #'var_vmin': 0,
            'var_cmap': mpl.cm.get_cmap('jet'),
            'saveFile': f'./midger_CalculateClimateMeans_{varName}_masked.pdf',
            #'dpi': 100,
            'figsize': (10, 4),
            }
    SLOTH.sloth.PlotLib.plot_ClimateYearMonth(org_data, data, **kwargs)

def find_largest_catchment(slopex, slopey, px_x, px_y):
    # initial indexes x=164, y=221
    # outlet index x=140, y=230
    max_nb_px = 0
    best_x = 0
    best_y = 0
    for i in range(px_x-5, px_x+5):
        for j in range(px_y-5, px_y+5):
    #for i in range(slopex.shape[1]):
    #    for j in range(slopey.shape[2]):
            catchment, nb_of_px, x, y = watershed_from_slopes(slopex, slopey, i, j)
            if nb_of_px > max_nb_px:
                max_nb_px = nb_of_px
                best_x = x
                best_y = y
    print("finished")
    print(f"number of pixels in the largest catchment: {max_nb_px}")
    print(f"x pixel index: {best_x}")
    print(f"y pixel index: {best_y}")
    return best_x, best_y

def one_point(x, y, matrix):
    for i, d1 in enumerate(matrix[0]):
        for j, d2 in enumerate(d1):
            if i==y and j==x:
                matrix[0][i][j] = 1
                print(f"only one point {x} and {y} have the value {matrix[0][i][j]}")
            else:
                matrix[0][i][j] = 0
    return matrix

def get_lon_lat(x=None, y=None):
    filepath = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/postpro/2020_01/TOT_PREC_ts.nc"
    ncfile = nc.Dataset(filepath)
    lon = ncfile["lon"][...]
    lat = ncfile["lat"][...]
    #print(lon[y][x])
    #print(lat[y][x])
    return lon, lat

def lon_lat_pixel(lon, lat):
    filepath = "/p/scratch/cslts/miaari1/TSMP_WorkflowStarter/simres/MainRun/1970010100/clm/clmoas.clm2.h0.1970-01-01-00900.nc"
    ncfile = nc.Dataset(filepath)
    alllon = ncfile["lon"][...]
    alllat = ncfile["lat"][...]
    print(alllon)
    min_lon = 100
    min_lat = 100
    i = 0
    for px_lon in alllon:
        diff = abs(px_lon-lon)
        if diff < min_lon:
            min_lon = diff
            print(diff)
            print(px_lon)
            print(lon)
            point_x = i
        i += 1
    j = 0
    for px_lat in alllat:
        diff = abs(px_lat-lat)
        if diff < min_lat:
            min_lat = diff
            point_y = j
        j += 1
    return point_x, point_y


def mask_slopes_watershed():
    slopexfile = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/geo/TSMP_EU11/static/parflow/slopex.sa"
    slopeyfile = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/geo/TSMP_EU11/static/parflow/slopey.sa"
    
    slopexfile = "/p/scratch/cslts/miaari1/TSMP_WorkflowStarter/geo/TSMP_EUR-11/static/parflow/EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_444x432_XSLOPE_TPS_HydroRIVER_sea_streams_corr.sa"
    slopeyfile = "/p/scratch/cslts/miaari1/TSMP_WorkflowStarter/geo/TSMP_EUR-11/static/parflow/EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_444x432_YSLOPE_TPS_HydroRIVER_sea_streams_corr.sa"
    
    slopex = SLOTH.sloth.IO.readSa(slopexfile)[0]
    slopey = SLOTH.sloth.IO.readSa(slopeyfile)[0]

    # Koeln
    lon = 6.96329
    lat = 50.93696
    # Lobith (Rhine)
    lon = 6.11 # x=197
    lat = 51.84 # y=234
    # Po (Ferrara)
    #lon = 11.6
    #lat = 44.88334
    # Nagymaros (Danube)
    #lon = 18.95
    #lat = 47.78
    # Regua (Porto)
    #lat = 41.154
    #lon = -7.782
    # Lobith (Neth)
    #lon = 6.11
    #lat = 51.84
    # Mozyr (Pol)
    #lon = 29.2667
    #lat = 52.05
    x, y = lon_lat_pixel(lon, lat)
#    x, y = get_lonlat_px_index(lon, lat)
    plt.imshow(slopex, origin="lower", interpolation="none")
    plt.scatter(x, y, color="red", marker="+", s=50)
    #plt.savefig("./x_y_plot.png")
    #return
    
    print(f"x is {x} and y is {y}")
    x, y = find_largest_catchment(slopex, slopey, x, y)
    catchment, nb_of_px, x, y  = watershed_from_slopes(slopex, slopey, x, y)
    #with open(f'./rhineriverbasin_slopes.npy', 'wb') as f:
    #    np.save(f, catchment)
    #catchment = one_point(x,y, catchment)
    plt.imshow(catchment, origin="lower", interpolation="none")
    plt.savefig("./x_y_plot.png")
    #masked = prudence_mask(catchment)
    #print(masked)
    #print(masked.shape)
    #org_data = get_temp_data()
    #test_plot(org_data, masked)
    #filepath = "/p/scratch/cslts/miaari1/python_test/maskedfile.tiff"
    #plt.imsave(filepath, masked)

def mask_by_shp():
    slopexfile = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/geo/TSMP_EU11/static/parflow/slopex.sa"
    slopeyfile = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/geo/TSMP_EU11/static/parflow/slopey.sa"
    slopex = SLOTH.sloth.IO.readSa(slopexfile)[0]
    slopey = SLOTH.sloth.IO.readSa(slopeyfile)[0]
    import geopandas as gpd
    from shapely.geometry import Point
    shp_path = "/p/scratch/cslts/miaari1/python_test/shp/rhine river basin/hybas_eu_lev04_v1c.shp"
    shpfile = gpd.read_file(shp_path)
    print(shpfile)

    shp_watershed = shpfile.geometry[0]
    all_lon, all_lat = get_lon_lat()
    catchment = np.zeros_like(all_lon)
    catchment = [catchment]
    
    for y in range(424):
        for x in range(436):
            print(y)
            lon = all_lon[y][x]
            lat = all_lat[y][x]
            p = Point(lon, lat)
            if shp_watershed.contains(p):
                catchment[0][y][x] = 1
            else:
                catchment[0][y][x] = 0
    #with open(f'./seineriverbasin.npy', 'wb') as f:
    #    np.save(f, catchment)
    x = 197
    y = 234
    plt.imshow(slopex, origin="lower", interpolation="none")
    plt.scatter(x, y, color="red", marker="+", s=50)
    plt.imshow(catchment[0], origin="lower", interpolation="none")
    plt.savefig("./catchment.png")
    
    
def catchment_from_point():
    slopexfile = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/geo/TSMP_EU11/static/parflow/slopex.sa"
    slopeyfile = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/geo/TSMP_EU11/static/parflow/slopey.sa"
    slopex = SLOTH.sloth.IO.readSa(slopexfile)[0]
    slopey = SLOTH.sloth.IO.readSa(slopeyfile)[0]
    print(slopex.shape)
    px_x = 199
    px_y = 212
    x, y = find_largest_catchment(slopex, slopey, px_x, px_y)
    catchment = SLOTH.sloth.toolBox.calc_catchment(slopex, slopey, x, y)
    plt.imshow(catchment, origin="lower", interpolation="none")
    plt.savefig("./catchment.png")


def plot_pressure_map():
    filepath = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/postpro/2020_01/press.nc"
    ncfile = nc.Dataset(filepath)
    pressure = ncfile["press"][...]
    for var in ncfile.variables:
        print(ncfile[var])
    
def surface_flow_px():
    x = 197
    y = 234
    all_year = {'day': [], 'month': [], 'River discharge (m3/s)': []}

    slopexfile = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/geo/TSMP_EU11/static/parflow/slopex.sa"
    slopeyfile = "/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/geo/TSMP_EU11/static/parflow/slopey.sa"
    slopex = SLOTH.sloth.IO.readSa(slopexfile)[0]
    slopey = SLOTH.sloth.IO.readSa(slopeyfile)[0]

    px_slope_x = slopex[y][x]
    px_slope_y = slopey[y][x]
    print(f"x direction slope: {px_slope_x}")
    print(f"y direction slope: {px_slope_y}")
    manning_coef = 5.5e-5 #hr/(m**/3)
    cell_width = 12500 #meter

    for i in range(1,12):
        i = f"{i}"
        i = i.zfill(2)
        print(i)
        filepath = f"/p/scratch/cslts/miaari1/FZJ-IBG3_WorkflowStarter/postpro/2020_{i}/press.nc"
        ncfile = nc.Dataset(filepath)
        pressure = ncfile["press"][...]
        px_timeseries = []

        for timestep in range(len(pressure[:-1])):
            px_press = pressure[timestep][-1][y][x]
            px_flow = ((abs(px_slope_y)**(0.5))/manning_coef)*(px_press**(5/3))*(1/3600)*cell_width
            px_timeseries.append(px_flow)
        
        day = 0
        for k in range(0, len(px_timeseries), 8):
            day += 1
            if i == '02' and day == 7:
                print("its hereeeeeeeeeeeeeeeeeeeeee")
                feb = px_timeseries
                febdf = pd.DataFrame(feb)
                febdf.to_csv('./feb_discharge.csv', index=False)
                return
                print(px_timeseries[k:k+8])
            all_year['day'].append(day)
            all_year['month'].append(i)
            daily_flow_y = np.mean(px_timeseries[k:k+8])
            #flow_y = ((abs(px_slope_y)**(0.5))/manning_coef)*(daily_press**(5/3))
            all_year['River discharge (m3/s)'].append(daily_flow_y)

    df = pd.DataFrame(all_year)
    df.to_csv("./rhineriver_discharge.csv", index=False)


def parflow_diagnostics():
    from ana_parflow_diagnostics_pythonheat.Diagnostics import Diagnostics
    import re
    # Ny ,Nx (number of grid points, integer)
    Nz = 15
    Ny = 432
    Nx = 444
    Dz = 15
    Dy = 432
    Dx = 444
    dir_path = "/p/scratch/cslts/miaari1/TSMP_WorkflowStarter/simres/MainRun/1970010100/parflow"
    files = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if re.findall("[0-9]",x.split("out.")[-1])]
    files.sort()

    # load mask data
    maskfilepath = "/p/scratch/cslts/miaari1/TSMP_WorkflowStarter/simres/MainRun/1970010100/parflow/ParFlow_EU11_1970010100.out.mask.pfb"
    mask = SLOTH.sloth.IO.read_pfb(maskfilepath)

    # slopex, slopey (2D slope, tensor float (1, ny, nx))
    slopexfile = "/p/scratch/cslts/miaari1/TSMP_WorkflowStarter/geo/TSMP_EUR-11/static/parflow/EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_444x432_XSLOPE_TPS_HydroRIVER_sea_streams_corr.sa"
    slopeyfile = "/p/scratch/cslts/miaari1/TSMP_WorkflowStarter/geo/TSMP_EUR-11/static/parflow/EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_444x432_YSLOPE_TPS_HydroRIVER_sea_streams_corr.sa"
    slopex = SLOTH.sloth.IO.readSa(slopexfile)
    slopey = SLOTH.sloth.IO.readSa(slopeyfile)

    # mannings (mannings n, tesnor float (1, ny, nx))
    manningfilepath = "/p/scratch/cslts/miaari1/TSMP_WorkflowStarter/simres/MainRun/1970010100/parflow/ParFlow_EU11_1970010100.out.n.pfb"
    mannings = SLOTH.sloth.IO.read_pfb(manningfilepath)[:1]

    # split=None (axis for parallel computation)
    split = None

    PD = Diagnostics(Mask=mask, Perm=None, Poro=None, Sstorage=None, Ssat=None, Sres=None, Nvg=None, Alpha=None, Mannings=mannings,
                    Slopex=slopex, Slopey=slopey, Dx=Dx, Dy=Dy, Dz=Dz, Dzmult=None, Nx=Nx, Ny=Ny, Nz=Nz, Terrainfollowing=None, Split=split)

    output = {"net lateral overland flow": []}
    i = 0
    for ncfilepath in files:
        print(i)
        i += 1
        #ncfilepath = "/p/scratch/cslts/miaari1/TSMP_WorkflowStarter/simres/MainRun/1970010100/parflow/ParFlow_EU11_1970010100.out.00001.nc"
        ncfile = nc.Dataset(ncfilepath)
        
        # top layer pressure
        pressure = ncfile["pressure"][...][0]        
        toplayerpress = PD.TopLayerPressure(Press=pressure)
        
        # overland fow
        flowx, flowy = PD.OverlandFlow(Toplayerpress=toplayerpress)
        #print(flowx)
        #print(flowy)
        # get flowx and flowy on a specific pixel
        # net lateral overland flow
        overlandflow = PD.NetLateralOverlandFlow(overland_flow_x=flowx, overland_flow_y=flowy)
        #print(overlandflow)
        #print(overlandflow.shape)
        #print("the pixel flowrate is:")
        maxitem = overlandflow[163][331]
        output["net lateral overland flow"].append(maxitem)
        #for item in overlandflow:
        #    for item2 in item:
        #        if item2 > maxitem:
        #            maxitem = item2
        #print(maxitem)
    
        df = pd.DataFrame(output)
        df.to_csv("/p/scratch/cslts/miaari1/python_test/overlandflow.csv", index=False)

def make_pfb():
    import ana_parflow_diagnostics_pythonheat.IO as io
    name = '/p/scratch/cslts/miaari1/ParFlow_scripts/pfdir/parflow/examples/outputs/default_richards/default_richards'
    split=None

    slopex   = io.read_pfb(name + '.out.specific_storage.pfb',split=split)
    
    for z in range(8):
        print(z)
        for y in range(10):
            for x in range(10):
                slopex[z][y][x] = 0.0005

    slopex_filename = '/p/scratch/cslts/miaari1/ParFlow_scripts/pfdir/parflow/examples/outputs/default_richards/default_richards.out.slope_x.pfb'
    slopey_filename = '/p/scratch/cslts/miaari1/ParFlow_scripts/pfdir/parflow/examples/outputs/default_richards/default_richards.out.slope_y.pfb'

    io.create_pfb(slopex_filename, slopex)
    io.create_pfb(slopey_filename, slopex)

    mannings_filename = '/p/scratch/cslts/miaari1/ParFlow_scripts/pfdir/parflow/examples/outputs/default_richards/default_richards.out.mannings.pfb'
    for z in range(8):
        print(z)
        for y in range(10):
            for x in range(10):
                slopex[z][y][x] = 2.3e-7
    io.create_pfb(mannings_filename, slopex)

def parflow_analysis():
    import ana_parflow_diagnostics_pythonheat.IO as io
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = "3all3top05slope"
    timesteps = ["0001", "0005"]
    pathname = f'/p/scratch/cslts/miaari1/ParFlow_scripts/pfdir/parflow/examples/outputs/{conditions}'
    for timestep in timesteps:
        print(f"timestep={timestep}")
        filename = os.path.join(pathname, f"default_richards.out.press.0{timestep}.pfb")
        layers = [0, -1]
        for layer in layers:
            layertitle = "Top" if layer==-1 else "Bottom"
            print(layertitle)
            split=None

            pressure = io.read_pfb(filename, split=split)
            print(pressure[layer])

            nrows = 10
            ncols = 10

            Z = np.array(pressure)
                    
            A = np.zeros((nrows,ncols))
            for x in range(0, nrows):
                for y in range(0, ncols):
                    a = np.float(Z[layer][x][y])
                    A[x, y] = a

            x = np.arange(ncols)
            y = np.arange(nrows)

            fig, ax = plt.subplots()
            ax.pcolormesh(x, y, A, shading='nearest', vmin=A.min(), vmax=A.max())


            def _annotate(ax, x, y, title):
                # this all gets repeated below:
                X, Y = np.meshgrid(x, y)
                ax.plot(X.flat, Y.flat, 'o', color='m')
                ax.set_xlim(-1, 11)
                ax.set_ylim(-1, 11)
                ax.set_title(title)
                plt.imshow(A)
                plt.colorbar()

                #plt.show()
                plt.savefig(f'/p/scratch/cslts/miaari1/ParFlow_scripts/pfdir/parflow/examples/outputs/figures/{conditions}_{layertitle}_{timestep}.png')

            _annotate(ax, x, y, f"{layertitle} layer")


def grdc_river_discharge():
    filepath1 = '/p/project/cslts/miaari1/python_scripts/grdc_river_discharge_jan-aug.txt'
    filepath2 = '/p/project/cslts/miaari1/python_scripts/grdc_river_discharge_aug-nov.txt'
    with open(filepath1) as f:
        lines = f.readlines()
    f.close()
    data = lines[0]
    data = data.replace(',','')
    discharge_values = re.findall("[\d]{3,5}.[0-9]{3,5}", data)
    discharge_values = [x for x in discharge_values if '2020' not in x]
    discharge_values.pop(59)
    discharge_values = discharge_values[:-3]

    jan_aug = discharge_values

    with open(filepath2) as f:
        lines = f.readlines()
    f.close()
    data = lines[0]
    data = data.replace(',','')
    discharge_values = re.findall("[\d]{3,5}.[0-9]{3,5}", data)
    discharge_values = [x for x in discharge_values if '2020' not in x]

    jan_aug.extend(discharge_values)

    print(jan_aug)
    print(len(jan_aug))

    dic = {'GRDC': jan_aug}
    df = pd.DataFrame(dic)
    df.to_csv('/p/project/cslts/miaari1/python_scripts/grdc.csv', index=False)
    print(df)

def plot_steady_state():

    z = list(range(1,41))
    z = [x/10 for x in z]
    dir_path = '/p/scratch/cslts/miaari1/ParFlow_scripts/pfdir/parflow/examples/tasks/steadystate_pressure'
    pfb_files = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x.endswith('.pfb') and 'ponding' not in x and "wt" not in x]
    x_list1 = []
    y_list1 = []
    x_list2 = []
    y_list2 = []
    x = []
    y = []
    for pfb_file in pfb_files:
        data = SLOTH.sloth.IO.read_pfb(pfb_file)
        filename = os.path.basename(pfb_file)
        time = filename.split('_')[-1].replace('.pfb','')
        q_k = filename.split('_')[2:4]
        q_k_mod = []
        

        for val in q_k:
            if val == '001':
                q_k_mod.append(0.001)
            if val == '01':
                q_k_mod.append(0.01)
            if val == '1':
                q_k_mod.append(0.1)
            if val == '10':
                q_k_mod.append(1.0)
            if val == '100':
                q_k_mod.append(10.0)
            if val == "2":
                q_k_mod.append(0.2)
            if val == "05":
                q_k_mod.append(0.05)
        
        #if (q_k_mod[0]==0.001 and q_k_mod[1]==0.001) or (q_k_mod[0]==0.1 and q_k_mod[1]==1) or (q_k_mod[0]==0.01 and q_k_mod[1]==1) or (q_k_mod[0]==0.001 and q_k_mod[1]==1):
        #    plt.plot(data[:,0,0],z, label=f'q={q_k_mod[0]} (m3/s), k={q_k_mod[1]} (m/s), steady state={time} (s)')
        #continue
        if len(q_k_mod) > 1:
            x.append(q_k_mod[1])
            y.append(time)
        continue
        q = q_k_mod
        if q_k_mod[0] == 0.001:
            #print(q_k_mod)
            q_k_mod = q_k_mod[0]/q_k_mod[1]
            #q = q_k_mod[0]
            #print(q_k_mod)
            q_k_mod = [q_k_mod for i in range(1,41)]
            #q = [q for i in range(1,41)]
            data = SLOTH.sloth.IO.read_pfb(pfb_file)
            # TODO check pressures at other pixels (expected to be the same)
            #plt.plot(data[:,0,0],z)
            plt.xscale("log")
            x_list1.append(q_k_mod[0])
            #y_list.append(data[:,0,0][-1])
            y_list1.append(int(time))
            #plt.plot(q_k_mod[0],data[:,0,0])
        if q[0] == 0.01:
            print(q)
            q = q[0]/q[1]
            #q = q_k_mod[0]
            #print(q_k_mod)
            q = [q for i in range(1,41)]
            #q = [q for i in range(1,41)]
            data = SLOTH.sloth.IO.read_pfb(pfb_file)
            # TODO check pressures at other pixels (expected to be the same)
            #plt.plot(data[:,0,0],z)
            plt.xscale("log")
            x_list2.append(q[0])
            #y_list.append(data[:,0,0][-1])
            y_list2.append(int(time))
    
    #dic = {'x': x_list1, 'y':y_list1}
    #df = pd.DataFrame(dic)
    #df.sort_values(by=['x'], ascending=True, inplace=True)
    #x_list1 = df['x']
    #y_list1 = df['y']
    dc = {'x': x, 'y':y}
    df1 = pd.DataFrame(dc)
    df1.sort_values(by=['x'], ascending=True, inplace=True)
    x_list2 = df1['x']
    y_list2 = df1['y']
    #plt.plot(x_list1,y_list1, label='q = 0.001 m3/s')
    #print(len(x_list2))
    #plt.plot(x_list2,y_list2, label='q = 0.01 m3/s')
    plt.plot(x_list2,y_list2)
    plt.xscale("log")
    plt.xlabel("q/k")
    plt.ylabel("Time to steady state (hr)")
    #plt.title("Pressure profile at steady state for various q and k")
    plt.legend()
    plt.show()

def pressure_steady_state():
    dir_path = '/p/scratch/cslts/miaari1/ParFlow_scripts/pfdir/parflow/examples/tasks/steadystate_pressure'
    pfb_files = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x.endswith('.pfb') and 'ponding' not in x and "wt" not in x]
    x_list = []
    y_list = []
    min_ax = [0]
    for pfb_file in pfb_files:
        data = SLOTH.sloth.IO.read_pfb(pfb_file)
        filename = os.path.basename(pfb_file)
        q_k = filename.split('_')[2:4]
        q_k_mod = []
        for val in q_k:
            if val == '001':
                q_k_mod.append(0.001)
            if val == '01':
                q_k_mod.append(0.01)
            if val == '1':
                q_k_mod.append(0.1)
            if val == '10':
                q_k_mod.append(1.0)
            if val == '100':
                q_k_mod.append(10.0)
        if (q_k_mod[0]==0.1 and q_k_mod[1]==1) or (q_k_mod[0]==0.01 and q_k_mod[1]==1) or (q_k_mod[0]==0.001 and q_k_mod[1]==1):
            q_k_mod = q_k_mod[0]/q_k_mod[1]
            #x_list.append(q_k_mod)
            #y_list.append(data[:,0,0][-1])
    #dic = {'x': x_list, 'y': y_list}
    #df = pd.DataFrame(dic)
    #df.sort_values(by=['x'], ascending=True, inplace=True)
    #x_list = df['x']
    #y_list = df['y']
            q_k_mod_list = [q_k_mod for x in range(40)]
            plt.plot(q_k_mod_list, data[:,0,0])
            min_ax = data[:,0,0] if min(data[:,0,0]) < min(min_ax) else min_ax
    plt.xscale("log")
    plt.xlabel("q/k")
    plt.ylim(0, -4)
    plt.ylabel("Pressure head (m)")
    #plt.title("Top layer pressure head at steady state")
    plt.show()

def exf_vs_inf():
    dir_path = '/p/scratch/cslts/miaari1/ParFlow_scripts/pfdir/parflow/examples/tasks/steadystate_pressure'
    pfb_files = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x.endswith('.pfb')]# and 'ponding' not in x]
    data = {'q':[], 'k':[], 'time_inf': []}
    for pfb_file in pfb_files:
        filename = os.path.basename(pfb_file)
        q_k = filename.split('_')[2:4]
        q_k_mod = []
        for val in q_k:
            if val == '001':
                q_k_mod.append(0.001)
            if val == '01':
                q_k_mod.append(0.01)
            if val == '1':
                q_k_mod.append(0.1)
            if val == '10':
                q_k_mod.append(1.0)
            if val == '100':
                q_k_mod.append(10.0)
        time = filename.split('_')[4].replace('.pfb','')
        inf_time = int(time)
        data["k"].append(q_k_mod[1])
        data["q"].append(q_k_mod[0])
        data["time_inf"].append(inf_time)
    df = pd.DataFrame(data)
    df['time_exf'] = 1
    dir_path = '/p/project/cslts/miaari1/figures/exfiltration'
    pfb_files = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x.endswith('.png')]# and 'ponding' not in x]
    for pfb_file in pfb_files:
        filename = os.path.basename(pfb_file)
        q_k = filename.split('_')[1:3]
        q_k_mod = []
        for val in q_k:
            if val == '001':
                q_k_mod.append(0.001)
            if val == '01':
                q_k_mod.append(0.01)
            if val == '1':
                q_k_mod.append(0.1)
            if val == '10':
                q_k_mod.append(1.0)
            if val == '100':
                q_k_mod.append(10.0)
        time = filename.split('_')[-1].replace('.png','')
        exf_time = int(time)
        print(exf_time)
        print(q_k_mod)
        if not df.loc[(df['q']==q_k_mod[0]) & (df['k']==q_k_mod[1]), 'time_exf'].empty:
            df.loc[(df['q']==q_k_mod[0]) & (df['k']==q_k_mod[1]), 'time_exf'] = exf_time

    #df["q/k"] = df["q"]/df["k"]
    #df["exf/inf"] = df["time_exf"]/df["time_inf"]
    
    df.sort_values(by=["q", "k"], ascending=True, inplace=True)
    df.to_csv('/p/project/cslts/miaari1/python_scripts/inf_exf.csv', index=False)
    print(df)
    return
    plt.plot(df["q/k"], df["exf/inf"])
    #plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel("q/k")
    plt.ylabel("Inf time/Exf time")
    plt.title("q/k agains time to reach infiltration steady state and fully dry")
    plt.show()

def plot_tables():
    df = pd.read_csv('/p/project/cslts/miaari1/python_scripts/kqtd.csv')
    print(df)
    x_axis = df["q/k"]
    dic = {"k": [0.001, 0.01], "d": [0.1, 0.2, 0.5]}
    for k in dic["k"]:
        for d in dic["d"]:
            if k==0.01 and d==0.5:
                continue
            y_axis = df[f"k{k}d{d}"]
            plt.plot(x_axis, y_axis, label=f"k={k}, d={4-d}")
    
    
    
    #plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel("q/k (-)")
    plt.ylabel("kt/d (-)")
    #plt.title("q/k vs qtda")
    plt.legend()
    plt.show()

def plot_parflow_scenarios():
    file_path = "/p/project/cslts/miaari1/python_scripts/all_settings_v2.csv"
    df = pd.read_csv(file_path)
    df["d"] = pd.to_numeric(df["d"])
    df["k"] = pd.to_numeric(df["k"])
    df["q/k"] = pd.to_numeric(df["q/k"])
    q_list = [0.0001, 0.001, 0.01, 0.1, 1]
    k_list = [0.0001, 0.001, 0.01, 0.1, 1]
    d_list = [3.5, 3, 2, 1]

    data = {"q/k": [0.0001, 0.001, 0.01, 0.1, 1]}
    for k in k_list:
        for d in d_list:
            print(f"k={k}, d={d}")
            temp_df = df.loc[(df["k"]==k) & (df["d"]==d)]
            data[f"k{k}d{d}"] = []
            for qk in data["q/k"]:
                qtd = temp_df.loc[df["q/k"]==qk, "exf_time"]
                if not qtd.empty:
                    data[f"k{k}d{d}"].append(qtd)
                    data["k"].append(k)
                else:
                    data[f"k{k}d{d}"].append(None)
                    data["k"].append(k)
    final_df = pd.DataFrame(data)
    columns = final_df.columns
    x_axis = final_df["q/k"]
    for k in k_list:
        for d in d_list:
            print(f"plotting k={k}, d={d}")
            y_axis = final_df[f"k{k}d{d}"]
            plt.plot(x_axis, y_axis, label=f"k={k}, d={4-d}")
    
    plt.xlabel("k (m/hr)")
    plt.ylabel("time (hr)")
    #plt.xscale("log")
    #plt.yscale("log")
    #plt.title("qt/k vs qtda")
    plt.legend()
    plt.show()

def plot_qk_qtd():
    file_path = "/p/project/cslts/miaari1/python_scripts/all_settings.csv"
    df = pd.read_csv(file_path)
    df["d"] = pd.to_numeric(df["d"])
    df["k"] = pd.to_numeric(df["k"])
    df["q/k"] = pd.to_numeric(df["q/k"])
    data = {"q/k": [0.0001, 0.001, 0.01, 0.1, 1]}
    q_list = [0.0001, 0.001, 0.01, 0.1, 1]
    k_list = [0.0001, 0.001, 0.01, 0.1, 1]
    d_list = [3.5, 3, 2, 1]
    df.sort_values(by=["q/k", "qt/d"], inplace=True, ascending=True)
    for qk in data["q/k"]:
        qtd = df.loc[df["q/k"]==qk, "qt/d"]
        qk_axis = [qk for i in range(len(qtd))]
        plt.plot(qk_axis, qtd)
    
    plt.xlabel("q/k (-)")
    plt.ylabel("qt/d (-)")
    plt.xscale("log")
    #plt.yscale("log")
    #plt.title("qt/k vs qtda")
    plt.legend()
    plt.show()

def prepare_exfiltration():
    dirpath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))), "scratch", "cslts", "miaari1", "ParFlow_scripts", "pfdir", "parflow", "examples", "tasks")
    file_path = "/p/project/cslts/miaari1/python_scripts/all_settings.csv"
    df = pd.read_csv(file_path)
    for i in range(len(df)):
        q = df["q"].iloc[i]
        k = df["k"].iloc[i]
        wt = df["wt"].iloc[i]
        if f"{q}".endswith("0"):
            q = f"{q}".replace(".0","")
        if f"{k}".endswith("0"):
            k = f"{k}".replace(".0","")
        if f"{wt}".endswith("0"):
            wt = f"{wt}".replace(".0","")
        
        foldername = f'exf_{q}_{k}_{wt}'
        new_path = os.path.join(dirpath, "watertable_exf", foldername)
        os.mkdir(new_path)
        time = df["time"].iloc[i]
        time = f"{time}"
        old_path = os.path.join(dirpath, "watertable", f'{q}_{k}_{wt}')
        shutil.copy(os.path.join(old_path, f"watertable.out.press.{time.zfill(5)}.pfb"), os.path.join(new_path, f'ic_{q}_{k}_{wt}_{time}.pfb'))
        shutil.copy(os.path.join(old_path, "watertable.tcl"), os.path.join(new_path, "watertable.tcl"))
        change_tcl_lines(os.path.join(new_path, "watertable.tcl"), "pfset Patch.top.BCPressure.alltime.Value", "0.0", "-")
        change_tcl_lines(os.path.join(new_path, "watertable.tcl"), "pfset TimingInfo.StopTime", "50000.0", "")
        change_tcl_lines(os.path.join(new_path, "watertable.tcl"), "pfset TimingInfo.DumpInterval", "100.0", "")


def change_tcl(filepath, text_to_replace, replace_with):    
    with open(filepath, 'r') as file :
        filedata = file.read()
    print(filedata)
    file.close()
    return
    # Replace the target string
    filedata = filedata.replace(text_to_replace, replace_with)

    # Write the file out again
    with open(filepath, 'w') as file:
      file.write(filedata)

def prepare_rerun_watertableinf():
    dirpath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))), "scratch", "cslts", "miaari1", "ParFlow_scripts", "pfdir", "parflow", "examples", "tasks", "watertable")
    folders = [os.path.join(dirpath, x) for x in os.listdir(dirpath)]
    for folder in folders:
        filepath = os.path.join(folder, "watertable.tcl")
        change_tcl(filepath, "100.0", "1.0")
        change_tcl(filepath, "40000.0", "10000.0")

def change_tcl_lines(filepath, fixed_statement, value, negative):
    old_file = open(filepath)
    lines = old_file.readlines()
    new_lines = []
    for line in lines:
        if fixed_statement in line:
            qvalue = re.findall("\d+\.\d+",line)
            print(qvalue)
            line = line.replace(f"{negative}{qvalue[0]}", value)
            print(line)
            new_lines.append(line)
        else:
            new_lines.append(line)
    old_file.close()
    with open(filepath, 'w') as f:
        f.write(''.join(new_lines))

def change_sh_file():
    dirpath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))), "scratch", "cslts", "miaari1", "ParFlow_scripts", "pfdir", "parflow", "examples", "tasks", "watertable_exf")
    folders = [os.path.join(dirpath, x) for x in os.listdir(dirpath)]
    for folder in folders:
        print(os.path.basename(folder))
        initial_cond = [x for x in os.listdir(folder) if "ic_" in x]
        filepath = os.path.join(folder, "watertable.tcl")
        file = open(filepath)
        lines = file.readlines()
        new_lines = []
        for line in lines:
            if "ic_pressure_10_100_1.pfb" in line:
                line = line.replace("ic_pressure_10_100_1.pfb", f"{initial_cond[0]}")
            new_lines.append(line)
        file.close()
        with open(filepath, 'w') as f:
            f.write(''.join(new_lines))

def steadystate_kinsol_drain():
    dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))), "scratch", "cslts", "miaari1", "ParFlow_scripts", "pfdir", "parflow", "examples", "tasks", "watertable_exf")
    scenarios = os.listdir(dir_path)
    df = pd.read_csv("/p/project/cslts/miaari1/python_scripts/all_settings.csv")
    df["exf_time"] = None
    for j in range(len(df)):
        q = df["q"].iloc[j]
        k = df["k"].iloc[j]
        wt = df["wt"].iloc[j]
        if f"{q}".endswith("0"):
            q = f"{q}".replace(".0","")
        if f"{k}".endswith("0"):
            k = f"{k}".replace(".0","")
        if f"{wt}".endswith("0"):
            wt = f"{wt}".replace(".0","")
        index = -1
        scen_dir = os.path.join(dir_path, f"exf_{q}_{k}_{wt}")
        print(f"exf_{q}_{k}_{wt}")
        kinsolfile = os.path.join(scen_dir, "watertable.out.kinsol.log")
        filedata = open(kinsolfile)
        lines = filedata.readlines()
        time_kinsol = [x for x in lines if "KINSOL starting step for time" in x or "KINSolInit nni=    0  fnorm=" in x]
        for i in range(1, len(time_kinsol)):
            if "KINSolInit nni=    0  fnorm=" not in time_kinsol[i] and "KINSolInit nni=    0  fnorm=" not in time_kinsol[i-1]:
                index = i-1
                time = re.findall("\d+\.\d+",time_kinsol[index])
                print(time)
                time = float(time[0])
                df["exf_time"].iloc[j] = time
                break

    print(df)
    df.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "all_settings_v2.csv"), index=False)

def plot_exftime():
    file_path = "/p/project/cslts/miaari1/python_scripts/all_settings_v2.csv"
    df = pd.read_csv(file_path)
    df["d"] = pd.to_numeric(df["d"])
    df["k"] = pd.to_numeric(df["k"])
    df["q/k"] = pd.to_numeric(df["q/k"])
    q_list = [0.0001, 0.001, 0.01, 0.1, 1]
    k_list = [0.0001, 0.001, 0.01, 0.1, 1]
    d_list = [3.5, 3, 2, 1]

    data = {"k": [0.0001, 0.001, 0.01, 0.1, 1]}
    for k in k_list:
        for d in d_list:
            print(f"k={k}, d={d}")
            temp_df = df.loc[(df["k"]==k) & (df["d"]==d)]
            data[f"k{k}d{d}"] = []
            print(len(temp_df["q/k"]))
            for qk in temp_df["q/k"]:
                qtd = temp_df.loc[df["t"]==qk, "exf_time"]
                if not qtd.empty:
                    if qtd.iloc[0]:
                        data[f"k{k}d{d}"].append(qtd.iloc[0])
                    else:
                        data[f"k{k}d{d}"].append(None)
                else:
                    data[f"k{k}d{d}"].append(None)
    print(data)
    final_df = pd.DataFrame(data)
    columns = final_df.columns
    x_axis = final_df["k"]
    for k in k_list:
        for d in d_list:
            print(f"plotting k={k}, d={d}")
            y_axis = final_df[f"k{k}d{d}"]
            plt.plot(x_axis, y_axis, label=f"k={k}, d={4-d}")
    
    plt.xlabel("k (m/hr)")
    plt.ylabel("time (hr)")
    #plt.xscale("log")
    #plt.yscale("log")
    #plt.title("qt/k vs qtda")
    plt.legend()
    plt.show()

def plot_roots():
    L = 400

    # Define the function
    def f(x):
        return np.tan(x*L) + 2*x

    # Create an array of x-values
    x_values = np.linspace(0, 10, 1000000)

    # Compute the corresponding y-values
    y_values = f(x_values)

    # Plot the function
    plt.plot(x_values, y_values, label='f(x) = tan(x*L) + 2x', color="black")

    # Add x-axis and y-axis lines
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

def find_roots_from_plot(L):

    # Define the function
    def f(x):
        y = np.tan(x*L) + 2*x
        return y

    # Create an array of x-values
    x_values = np.linspace(0, 20, 1000000)
    print("calculating y values")
    # Compute the corresponding y-values
    y_values = f(x_values)

    zeros = [0] * len(x_values)
    line1 = LineString(np.column_stack((x_values, y_values)))
    line2 = LineString(np.column_stack((x_values, zeros)))
    print("looking for intersections")
    intersection = line1.intersection(line2)

    points = [p.x for p in intersection]
    roots = [point for point in points if abs(f(point))< 0.0001]


    return roots
    plt.plot(x_values, y_values, color="black")
    plt.plot(x_values, zeros, color="black")
    new_zeros = [0 for i in range(len(points))]
    plt.plot(points, new_zeros, "ro")
    plt.xlabel("λ")
    plt.ylabel("f(λ)")
    plt.show()


def newton_raphson_func(x0, L):
    #L = 1
    tolerance = 1e-8  # Set the desired tolerance for convergence
    max_iterations = 1000  # Set a maximum number of iterations

    # Define the function f(x) and its derivative f'(x)
    def f(x):
        return np.tan(x*L) + 2*x

    def df(x):
        return L*(1/np.cos(x*L)**2) + 2

    # Implement the Newton-Raphson method
    def newton_raphson(x0):
        x_n = x0
        for _ in range(max_iterations):
            f_value = f(x_n)
            df_value = df(x_n)
            x_n1 = x_n - f_value / df_value

            if abs(x_n1 - x_n) < tolerance:
                return x_n1
            x_n = x_n1

        return None  # If the method does not converge within the maximum iterations

    # Initial guess for the root
    #x0 = 100

    # Find the root using the Newton-Raphson method
    root = newton_raphson(x0)

    if root is not None:
        #print("Approximate root:", root)
        return root
    else:
        #print("The Newton-Raphson method did not converge to a root.")
        return None

def residue(lambdaa, L, t, z):
    residue_value = (np.sin(lambdaa*z))*np.sin(lambdaa*L)*(np.exp(-t*(lambdaa*lambdaa)))/(1+(L/2)+(2*L*lambdaa*lambdaa))
    return residue_value

def sum_residue(L, t, z, roots):
    sum_values = []
    #plot_list = []
    
    for root in roots:
        resid = residue(root, L, t, z)
        sum_values.append(resid)
        
        #if z>0:
        #    plot_list.append(np.sum(sum_values))
    #if z >0:
    #    plt.plot(roots, plot_list, color="black")
    #    plt.xlabel("λ")
    #    plt.ylabel("Sum of residue")
    #    plt.show()
    #    return
    
    residues = np.sum(sum_values)

    return residues

def analytical_pressure():
    # define variables in cm and hr
    #L_star = 100 #cm
    L_star = 400 #cm
    Ks = 1 #cm/hr
    #alfa = 0.1 # 1/cm
    alfa = 0.02 # NOT 1/cm but fitted based on K from Van Genuchten
    #Ss = 0.4
    Ss = 1
    #Sr = 0.06
    Sr = 0.2
    #qa_star = 0.1 #cm/hr
    qa_star = 0 #cm/hr
    #qb_star = 0.9 #cm/hr
    qb_star = 0.1 #cm/hr
    pressure_0 = 0 #cm

    # define dimensionless parameters
    L = alfa*L_star
    qa = qa_star/Ks
    qb = qb_star/Ks

    # get the roots for this L
    roots = find_roots_from_plot(L)

    # define layers
    data = {}
    data = {"z": [*range(0, L_star+1, 1)]}
    # calculate for timesteps
    timesteps = range(1, 2002, 100)
    #timesteps = [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    #timesteps = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]
    for t_star in timesteps:
        data[f"time={t_star}"] = []
        t = (alfa*Ks*t_star)/(Ss-Sr)
        for z_star in data["z"]:
            #plotit = True
            
            # for z_star=0 bottom layer, z_star=L_star top layer
            z = alfa*z_star
            #if z>0:
            #    plotit = False
            residues = sum_residue(L, t, z, roots)
            #if not plotit:
            #    print(residues)
            #    return
            #print(f"for t_star={t_star} and z={z_star} the residue is: {residues}")
            
            K = qb - (qb - np.exp(alfa*pressure_0))*np.exp(-z) - 4*(qb - qa)*(np.exp((L-z)/2)*np.exp(-t/4))*residues

            K_star = K*Ks
            pressure = (np.log(K_star/Ks))/alfa
            data[f"time={t_star}"].append(pressure)
            
    
    df = pd.DataFrame(data)
    print(df)
    #df.to_csv("/p/project/cslts/miaari1/python_scripts/analytical_pressure.csv", index=False)
    count = -1
    for time in timesteps:
        x_axis = df[f"time={time}"].to_list()
        y_axis = df["z"].to_list()
        plt.plot(x_axis, y_axis, label=f"t={time}")
        plt.annotate(f"{time}", xy=(x_axis[count], y_axis[count]))
        count -= 8
    plt.xlabel("pressure head (cm)")
    plt.ylabel("z (cm)")
    plt.legend()
    plt.show()


def prepare_plotsfrom_analytical():
    plt.rcParams.update({'font.size': 14})
    # L = 400 cm
    # Ks = 1 cm/hr
    # time at steady state = 2000 ?
    # alfa = 0.01 /cm
    # Ss = 1
    # Sr = 0.2
    # q = 0.1 cm/hr

    # define variables in cm and hr
    #L_star = 100 #cm
    L_star = 400 #cm
    Ks = 1 #cm/hr
    #alfa = 0.1 # 1/cm
    alfa = 0.01 # 1/cm
    #Ss = 0.4
    Ss = 1
    #Sr = 0.06
    Sr = 0.2
    qa_star = 0.0 #cm/hr
    
    qb_star = 0.0 #cm/hr
    pressure_0 = 0 #cm

    # define dimensionless parameters
    L = alfa*L_star
    qa = qa_star/Ks
    qb = qb_star/Ks

    # get the roots for this L
    roots = find_roots_from_plot(L)
    print(roots)
    def f(x):
        y = np.tan(x*L) + 2*x
        return y
    for x in roots:
        print(f(x))
    #roots_df = pd.DataFrame({"roots": roots})
    #roots_df.to_csv("/p/project/cslts/miaari1/python_scripts/roots_100cm10.csv", index=False)
    #roots_df = pd.read_csv("/p/project/cslts/miaari1/python_scripts/roots_100cm.csv")
    #roots = roots_df["roots"].to_list()   

    # define layers
    data = {"z": [*range(5, L_star+1, 10)]}
    # calculate for timesteps
    #timesteps = range(1, 100, 10)
    #timesteps = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    timesteps = [100]
    for t_star in timesteps:
        data[f"time={t_star}"] = []
        t = (alfa*Ks*t_star)/(Ss-Sr)
        for z_star in data["z"]:
            # for z_star=0 bottom layer, z_star=L_star top layer
            z = alfa*z_star
            residues = sum_residue(L, t, z, roots)
            #if z > 1:
            #    print(residues)
            #    return
            #print(f"for t_star={t_star} and z={z_star} the residue is: {residues}")
            
            K = qb - (qb - np.exp(alfa*pressure_0))*np.exp(-z) - 4*(qb - qa)*(np.exp((L-z)/2)*np.exp(-t/4))*residues
            #print(f"the value of k: {K}")
            K_star = K*Ks
            pressure = (np.log(K_star/Ks))/alfa
            data[f"time={t_star}"].append(pressure)
    
    df = pd.DataFrame(data)
    print(df)
    df = df*0.01
    #df.to_csv("/p/project/cslts/miaari1/python_scripts/analytical_pressure.csv", index=False)
    count = -1
    for time in timesteps:
        x_axis = df[f"time={time}"].to_list()
        y_axis = df["z"].to_list()
        plt.plot(x_axis, y_axis, label=f"t=0",  color='black')
        plt.annotate(f"t=0", xy=(x_axis[-1], y_axis[-1]))
        count -= 4




    # define variables in cm and hr
    n = 2
    m = 1-1/n
    #L_star = 100 #cm
    L_star = 400 #cm
    Ks = 1 #cm/hr
    #alfa = 0.1 # 1/cm
    alfa = 0.01 # 1/cm
    #Ss = 0.4
    Ss = 1
    #Sr = 0.06
    Sr = 0.2
    #qa_star = 0.1 #cm/hr
    qa_star = 0 #cm/hr
    #qb_star = 0.9 #cm/hr
    qb_star = 0.1 #cm/hr
    pressure_0 = 0 #cm

    # define dimensionless parameters
    L = alfa*L_star
    qa = qa_star/Ks
    qb = qb_star/Ks

    # get the roots for this L
    roots = find_roots_from_plot(L)
    print(roots)
    def f(x):
        y = np.tan(x*L) + 2*x
        return y
    for x in roots:
        print(f(x))
    #roots_df = pd.DataFrame({"roots": roots})
    #roots_df.to_csv("/p/project/cslts/miaari1/python_scripts/roots_100cm10.csv", index=False)
    #roots_df = pd.read_csv("/p/project/cslts/miaari1/python_scripts/roots_100cm.csv")
    #roots = roots_df["roots"].to_list()   

    # define layers
    data = {}
    data = {"z": [*range(5, L_star+1, 10)]}
    # calculate for timesteps
    timesteps = range(1, 2001, 1)
    #timesteps = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    #timesteps = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]
    for t_star in timesteps:
        data[f"time={t_star}"] = []
        t = (alfa*Ks*t_star)/(Ss-Sr)
        for z_star in data["z"]:
            # for z_star=0 bottom layer, z_star=L_star top layer
            z = alfa*z_star
            residues = sum_residue(L, t, z, roots)
            #if z > 1:
            #    print(residues)
            #    return
            #print(f"for t_star={t_star} and z={z_star} the residue is: {residues}")
            
            K = qb - (qb - np.exp(alfa*pressure_0))*np.exp(-z) - 4*(qb - qa)*(np.exp((L-z)/2)*np.exp(-t/4))*residues
            #print(f"the value of k: {K}")
            K_star = K*Ks
            pressure = (np.log(K_star/Ks))/alfa
            data[f"time={t_star}"].append(pressure)
    
    df = pd.DataFrame(data)
    print(df)
    df = df*0.01
    print(df)
    #df.to_csv("/p/project/cslts/miaari1/python_scripts/analytical_pressure.csv", index=False)
    count = -1
    timesteps = [1, 10, 100, 1091]
    #timesteps = [*range(1091, 1999, 1)]
    for time in timesteps:
        x_axis = df[f"time={time}"].to_list()
        y_axis = df["z"].to_list()
        plt.plot(x_axis, y_axis, label=f"t={time}", color='blue')
        plt.annotate(f"{time}", xy=(x_axis[count], y_axis[count]), color="blue")
        count -= 4


    # PLOTTING FROM PARFLOW
    name='/p/project/cslts/miaari1/python_scripts/parflow/comparewithanalytical/infiltration/infiltration'
    #z = list(range(0,401))
    nt=2000
    alldata = {}
    count = -1

    for t in timesteps:
        #t = 193
        print(name + '.out.satur.'+('{:05d}'.format(t))+'.pfb')
        #data = pfr.read(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
        data = SLOTH.sloth.IO.read_pfb(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
        x_axis = data[:,0,0]

        plt.plot(x_axis, y_axis, label=f"t={t}", color='red')
        plt.annotate(f"{t}", xy=(x_axis[count], y_axis[count]), color="red")
        count -= 4

    plt.xlabel("pressure head (m)")
    plt.ylabel("z (m)")
    line_up, = plt.plot([], label='Analytical solution', color="blue")
    line_down, = plt.plot([], label='Numerical solution', color="red")
    plt.legend(handles=[line_up, line_down])
    plt.show()


def prepare_plotsfrom_analytical_yehexample():
    plt.rcParams.update({'font.size': 16})
    # L = 400 cm
    # Ks = 1 cm/hr
    # time at steady state = 2000 ?
    # alfa = 0.01 /cm
    # Ss = 1
    # Sr = 0.2
    # q = 0.1 cm/hr

    # define variables in cm and hr
    L_star = 100 #cm
    #L_star = 400 #cm
    Ks = 1 #cm/hr
    alfa = 0.1 # 1/cm
    #alfa = 0.01 # 1/cm
    Ss = 0.4
    #Ss = 1
    Sr = 0.06
    #Sr = 0.2
    qa_star = 0.0 #cm/hr
    
    qb_star = 0.1 #cm/hr
    pressure_0 = 0 #cm

    # define dimensionless parameters
    L = alfa*L_star
    qa = qa_star/Ks
    qb = qb_star/Ks

    # get the roots for this L
    roots = find_roots_from_plot(L)
    print(roots)
    def f(x):
        y = np.tan(x*L) + 2*x
        return y
    for x in roots:
        print(f(x))
    #roots_df = pd.DataFrame({"roots": roots})
    #roots_df.to_csv("/p/project/cslts/miaari1/python_scripts/roots_100cm10.csv", index=False)
    #roots_df = pd.read_csv("/p/project/cslts/miaari1/python_scripts/roots_100cm.csv")
    #roots = roots_df["roots"].to_list()   

    # define layers
    data = {"z": [*range(5, L_star+1, 10)]}
    # calculate for timesteps
    #timesteps = range(1, 100, 10)
    #timesteps = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    timesteps = [1000]
    for t_star in timesteps:
        data[f"time={t_star}"] = []
        t = (alfa*Ks*t_star)/(Ss-Sr)
        for z_star in data["z"]:
            # for z_star=0 bottom layer, z_star=L_star top layer
            z = alfa*z_star
            residues = sum_residue(L, t, z, roots)
            #if z > 1:
            #    print(residues)
            #    return
            #print(f"for t_star={t_star} and z={z_star} the residue is: {residues}")
            
            K = qb - (qb - np.exp(alfa*pressure_0))*np.exp(-z) - 4*(qb - qa)*(np.exp((L-z)/2)*np.exp(-t/4))*residues
            #print(f"the value of k: {K}")
            K_star = K*Ks
            pressure = (np.log(K_star/Ks))/alfa
            data[f"time={t_star}"].append(pressure)
    
    df = pd.DataFrame(data)
    print(df)
    df = df*0.01
    #df.to_csv("/p/project/cslts/miaari1/python_scripts/analytical_pressure.csv", index=False)
    count = -1
    for time in timesteps:
        x_axis = df[f"time={time}"].to_list()
        y_axis = df["z"].to_list()
        plt.plot(x_axis, y_axis, label=f"t=0",  color='blue')
        plt.annotate(f"t=0", xy=(x_axis[-1], y_axis[-1]), color="blue")
        count -= 4




    # define variables in cm and hr
    L_star = 100 #cm
    #L_star = 400 #cm
    Ks = 1 #cm/hr
    #alfa = 0.1 # 1/cm
    alfa = 0.1 # 1/cm
    Ss = 0.4
    #Ss = 1
    Sr = 0.06
    #Sr = 0.2
    qa_star = 0.1 #cm/hr
    #qa_star = 0 #cm/hr
    qb_star = 0.9 #cm/hr
    #qb_star = 0.1 #cm/hr
    pressure_0 = 0 #cm

    # define dimensionless parameters
    L = alfa*L_star
    qa = qa_star/Ks
    qb = qb_star/Ks

    # get the roots for this L
    roots = find_roots_from_plot(L)
    print(roots)
    def f(x):
        y = np.tan(x*L) + 2*x
        return y
    for x in roots:
        print(f(x))
    #roots_df = pd.DataFrame({"roots": roots})
    #roots_df.to_csv("/p/project/cslts/miaari1/python_scripts/roots_100cm10.csv", index=False)
    #roots_df = pd.read_csv("/p/project/cslts/miaari1/python_scripts/roots_100cm.csv")
    #roots = roots_df["roots"].to_list()   

    # define layers
    data = {}
    data = {"z": [*range(5, L_star+1, 10)]}
    # calculate for timesteps
    timesteps = range(1, 2001, 1)
    #timesteps = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    #timesteps = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]
    for t_star in timesteps:
        data[f"time={t_star}"] = []
        t = (alfa*Ks*t_star)/(Ss-Sr)
        for z_star in data["z"]:
            # for z_star=0 bottom layer, z_star=L_star top layer
            z = alfa*z_star
            residues = sum_residue(L, t, z, roots)
            #if z > 1:
            #    print(residues)
            #    return
            #print(f"for t_star={t_star} and z={z_star} the residue is: {residues}")
            
            K = qb - (qb - np.exp(alfa*pressure_0))*np.exp(-z) - 4*(qb - qa)*(np.exp((L-z)/2)*np.exp(-t/4))*residues
            #print(f"the value of k: {K}")
            K_star = K*Ks
            pressure = (np.log(K_star/Ks))/alfa
            data[f"time={t_star}"].append(pressure)
    
    df = pd.DataFrame(data)
    print(df)
    df = df*0.01
    print(df)
    df.to_csv("/p/project/cslts/miaari1/python_scripts/analytical_pressure_yehexample.csv", index=False)
    count = -1
    timesteps = [1, 10, 50, 75]
    #timesteps = [*range(1091, 1999, 1)]
    for time in timesteps:
        x_axis = df[f"time={time}"].to_list()
        y_axis = df["z"].to_list()
        plt.plot(x_axis, y_axis, label=f"t={time}", color='blue')
        plt.annotate(f"{time}", xy=(x_axis[count], y_axis[count]), color="blue")
        count -= 1



    # PLOTTING FROM PARFLOW
    name='/p/project/cslts/miaari1/python_scripts/parflow/comparewithanalytical/srivastavaandyeh_0-01_final/exfiltration'
    #z = list(range(0,401))
    nt=2000
    alldata = {}
    count = -1
    timesteps = [0, 1, 10, 50, 75]

    for t in timesteps:
        #t = 193
        print(name + '.out.satur.'+('{:05d}'.format(t))+'.pfb')
        #data = pfr.read(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
        data = SLOTH.sloth.IO.read_pfb(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
        x_axis = data[:,0,0]

        plt.plot(x_axis, y_axis, label=f"t={t}", color='red')
        plt.annotate(f"{t}", xy=(x_axis[count], y_axis[count]), color="red")
        count -= 2

    plt.xlabel("pressure head (m)")
    plt.ylabel("z (m)")
    line_up, = plt.plot([], label='Analytical solution', color="blue")
    line_down, = plt.plot([], label='Numerical solution', color="red")
    plt.legend(handles=[line_up, line_down])
    plt.show()

def parflow_to_csv():
    name='/p/project/cslts/miaari1/python_scripts/parflow/comparewithanalytical/infiltration_2/infiltration'
    z = list(range(5, 401, 10))

    nt=2000
    timesteps = [0, 1, 10, 100, 1091]
    timesteps = list(range(0,15,1))
    alldata = {"z": z}
    for t in timesteps:
        #t = 193
        print(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
        #data = pfr.read(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
        data = SLOTH.sloth.IO.read_pfb(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
        alldata[f"time={t}"] = data[:,0,0]

    #print(alldata)
    df = pd.DataFrame(alldata)
    df["z"] = df["z"]*0.01
    #print(df)
    df.to_csv("/p/project/cslts/miaari1/python_scripts/outputs/numerical_pressure_2.csv", index=False)

def prepare_plotsfrom_analytical_drainage():
    plt.rcParams.update({'font.size': 14})
    
    # define variables in cm and hr
    #L_star = 100 #cm
    L_star = 400 #cm
    Ks = 1 #cm/hr
    #alfa = 0.1 # 1/cm
    alfa = 0.01 # 1/cm
    #Ss = 0.4
    Ss = 1
    #Sr = 0.06
    Sr = 0.2
    #qa_star = 0.1 #cm/hr
    qa_star = 0.1 #cm/hr
    #qb_star = 0.9 #cm/hr
    qb_star = 0 #cm/hr
    pressure_0 = 0 #cm

    # define dimensionless parameters
    L = alfa*L_star
    qa = qa_star/Ks
    qb = qb_star/Ks

    # get the roots for this L
    roots = find_roots_from_plot(L)
    print(roots)
    def f(x):
        y = np.tan(x*L) + 2*x
        return y
    for x in roots:
        print(f(x))
    #roots_df = pd.DataFrame({"roots": roots})
    #roots_df.to_csv("/p/project/cslts/miaari1/python_scripts/roots_100cm10.csv", index=False)
    #roots_df = pd.read_csv("/p/project/cslts/miaari1/python_scripts/roots_100cm.csv")
    #roots = roots_df["roots"].to_list()   

    # define layers
    data = {}
    data = {"z": [*range(5, L_star+1, 10)]}
    # calculate for timesteps
    timesteps = range(1, 20001, 1)
    #timesteps = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    #timesteps = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]
    for t_star in timesteps:
        data[f"time={t_star}"] = []
        t = (alfa*Ks*t_star)/(Ss-Sr)
        for z_star in data["z"]:
            # for z_star=0 bottom layer, z_star=L_star top layer
            z = alfa*z_star
            residues = sum_residue(L, t, z, roots)
            #if z > 1:
            #    print(residues)
            #    return
            #print(f"for t_star={t_star} and z={z_star} the residue is: {residues}")
            
            K = qb - (qb - np.exp(alfa*pressure_0))*np.exp(-z) - 4*(qb - qa)*(np.exp((L-z)/2)*np.exp(-t/4))*residues
            #print(f"the value of k: {K}")
            K_star = K*Ks
            pressure = (np.log(K_star/Ks))/alfa
            data[f"time={t_star}"].append(pressure)
    
    df = pd.DataFrame(data)
    print(df)
    df = df*0.01
    print(df)
    df.to_csv("/p/project/cslts/miaari1/python_scripts/analytical_pressure_drainage.csv", index=False)
    count = -1
    timesteps = [100, 1000, 10000, 19600]
    #timesteps = [*range(1091, 1999, 1)]
    for time in timesteps:
        x_axis = df[f"time={time}"].to_list()
        y_axis = df["z"].to_list()
        plt.plot(x_axis, y_axis, label=f"t={time}", color='blue')
        plt.annotate(f"t={time}", xy=(x_axis[count], y_axis[count]), color="blue")
        count -= 4


    # PLOTTING FROM PARFLOW
    name='/p/project/cslts/miaari1/python_scripts/parflow/comparewithanalytical/exfiltration/exfiltration'
    #z = list(range(0,401))
    nt=2000
    alldata = {}
    count = -1
    timesteps = [1, 10, 100, 196]

    for t in timesteps:
        #t = 193
        print(name + '.out.satur.'+('{:05d}'.format(t))+'.pfb')
        #data = pfr.read(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
        data = SLOTH.sloth.IO.read_pfb(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
        x_axis = data[:,0,0]

        plt.plot(x_axis, y_axis, label=f"t={t}", color='red')
        plt.annotate(f"t={t*100}", xy=(x_axis[count], y_axis[count]), color="red")
        count -= 8

    plt.xlabel("pressure head (m)")
    plt.ylabel("z (m)")
    line_up, = plt.plot([], label='Analytical solution', color="blue")
    line_down, = plt.plot([], label='Numerical solution', color="red")
    plt.legend(handles=[line_up, line_down])
    plt.show()


def save_firsttimesteps():
    name='/p/scratch/cslts/miaari1/ParFlow_scripts/pfdir/parflow/examples/tasks/watertable/0.01_0.1_1/watertable'
    z = list(range(1,41))
    
    nt=5
    dic = {"z": z}
    for t in range (nt+1):
        #t = 193
        print(name + '.out.satur.'+('{:05d}'.format(t))+'.pfb')
        #data = pfr.read(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
        data = SLOTH.sloth.IO.read_pfb(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
        dic[f"t={t}"] = data[:,0,0]
        #plt.plot(data[:,0,0],z)
    
    df = pd.DataFrame(dic)
    df["z"] = df["z"]*0.1
    df.to_csv("/p/project/cslts/miaari1/python_scripts/firsttimesteps.csv", index=False)


def plot_qk_ss():
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score, mean_absolute_error
    import math

    def powerlaw_fit(h, a, b):
        y = a*(h**b)
        #y = (a)*np.exp(-b*h)
        return y
    plt.rcParams.update({'font.size': 22})
    #filepath = '/p/project/cslts/miaari1/python_scripts/outputs/testcases_appended.csv' # with watertable depth, n, alfa, theta, Ks from rosetta
    #filepath = '/p/project/cslts/miaari1/python_scripts/outputs/all_settings_v2.csv' # with watertable depth
    #filepath = '/p/project/cslts/miaari1/python_scripts/outputs/qk_ss.csv' # without watertable depth
    #filepath = '/p/project/cslts/miaari1/python_scripts/outputs/testcases_constantq.csv' # without watertable depth
    filepath = '/p/project/cslts/miaari1/python_scripts/outputs/testcases_constantq_constantd.csv' # without watertable depth

    q_column = "q"
    k_column = "k"
    d_column = "d"
    n_column = "n"
    alfa_column = "alfa"
    theta_r_column = "theta_r"
    theta_s_column = "theta_s"
    t_column = "time"
    df = pd.read_csv(filepath)
    print(df)
    #indexes = list(range(1165, 1215))
    #df = df[~1165:~1215]
    #df = df.drop(indexes, axis=0)
    #df.reset_index(inplace=True, drop=True)
    #print(df)

    #d = 4 #4m watertable depth
    #L = 4 # not accounting for the vertical profile, for d=3m then L=3m

    #alfa = 1
    #n = 2
    #theta_s = 1
    #theta_r = 0.2
    t_list = []
    x_axis = []
    y_axis = []
    for i in range(len(df)):
        q = df[q_column].iloc[i]
        k = df[k_column].iloc[i]
        d = df[d_column].iloc[i]
        n = df[n_column].iloc[i]
        alfa = df[alfa_column].iloc[i]
        theta_r = df[theta_r_column].iloc[i]
        theta_s = df[theta_s_column].iloc[i]
        t = df[t_column].iloc[i]
        if k>=q and t!=0:# and (q/k)>=0.001:
            #q = q/Ks
            #k = k/Ks
            #d = alfa*d
            #t = (alfa*k*t)/(theta_s-theta_r)
            #L = alfa*d
            
            x = k
            #x = (alfa*d*q/k)
            #y = (k*k*t/(q*d*d*alfa))**(1-1/n) # best performance
            #y = (k*t/(d*d*alfa))**(1-1/n)
            #y = (q*q*t/(k*alfa*d*d))**(1-1/n)
            #y = (q*d*d*alfa/(k*k*t))**(1-1/n) # best fit
            y = t#*k/d
            #y = (k*t/d)#**(1-1/n)
            #y = alfa*d*t*k*alfa/(theta_s-theta_r)#*q/d)

            #y_pred = 1.9886875606876062*(x**(-0.5468849399415959))
            #t_pred = (y_pred*y_pred*q*alfa*d*d)/(k*k)
            #t_list.append(abs(t_pred-t))
            y_axis.append(y)
            x_axis.append(x)

    params, covariance = curve_fit(powerlaw_fit, x_axis, y_axis)
    # Extract the fitted parameters
    a_fit, b_fit = params

    # Print the results
    print(f"Fitted a: {a_fit} and b:{b_fit}")

    # Generate data points for the fitted curve
    y_fit = [powerlaw_fit(x, a_fit, b_fit) for x in x_axis]

    R_square = r2_score(y_axis, y_fit)
    print(f"R2 = {R_square}")
    #return R_square
    MSE = np.square(np.subtract(y_axis,y_fit)).mean()
    print(f"MSE = {MSE*10**-9} x10⁹")
    RMSE = math.sqrt(MSE)
    print(f"RMSE = {RMSE}")
    MAE = mean_absolute_error(y_axis, y_fit)
    print(f"MAE = {MAE}")

    # calculate difference in t
    #print(t_list)
    #print(f"avg t = {np.mean(t_list)}")
    #print(f"max t = {np.max(t_list)} min t = {np.min(t_list)}")
    #plt.scatter(x_axis, t_list, s=30, facecolors='r', edgecolors='r')
    
    #org_bin_centers, org_counts = plot_pdf(y_axis)
    #fit_bin_centers, fit_counts = plot_pdf(y_fit)
    #plt.plot(org_bin_centers, org_counts, linestyle='-', label="Simulations")#, marker='o')
    #plt.plot(fit_bin_centers, fit_counts, linestyle='-', label="Fitted powerlaw")#, marker='o')
    #plt.xlabel('(qαd²/k²t)^(1-1/n) (-)')
    #plt.ylabel('PDF')
    #plt.title('Probability Density Function')
    #plt.legend()
    #plt.grid(True)
    #plt.show()
    
    x_fit = [min(x_axis), max(x_axis)]
    y_fit = [powerlaw_fit(x, a_fit, b_fit) for x in x_fit]

    plt.figure(figsize=(16,9))
    plt.grid(True)
    #plt.plot(x_axis, libreoffice_fit, color="green", label="libreoffice")
    plt.scatter(x_axis, y_axis, s=30, facecolors='k', edgecolors='k', label="data points")
    #plt.plot(x_fit, y_fit, color="black", label="fitted line")
    #plt.scatter(x_axis, y_fit, s=30, facecolors='b', edgecolors='b')# color="black", label="fitted")
    #plt.annotate(f"R²={round(R_square, 3)}\nf(x)={round(a_fit, 3)}x^({round(b_fit, 3)})", xy=(10, 0.1), color="black")
    # α
    plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel("Ks (m/hr)")
    plt.ylabel("t (hr)")
    plt.title("q=0.0002m/hr, d=4m")
    plt.legend()
    plt.show()

    #y_fit = [powerlaw_fit(x, a_fit, b_fit) for x in x_axis]
    #plt.scatter(y_axis, y_fit, s=30, facecolors='r', edgecolors='r', label="data points")
    #plt.plot([0,100], [0, 100], 'k-')
    #plt.xlim([0.01, 100])
    #plt.ylim([0.01, 100])
    #plt.xscale("log")
    #plt.yscale("log")
    #plt.xlabel("Simulations")
    #plt.ylabel("Fitted powerlaw")
    #plt.show()

def plot_qk_ss_drainage():
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score, mean_absolute_error
    import math
    print("drainage case")
    def powerlaw_fit(h, a, b):
        y = (a)*(h**b)
        # decay function
        #y = (a)*np.exp(-b*h)
        return y
    plt.rcParams.update({'font.size': 22})
    filepath = '/p/project/cslts/miaari1/python_scripts/outputs/all_settings_v2.csv' # with watertable depth
    #filepath = '/p/project/cslts/miaari1/python_scripts/outputs/qk_ss.csv' # without watertable depth
    
    q_column = "q"
    k_column = "k"
    d_column = "d"
    t_column = "exf_time"
    df = pd.read_csv(filepath)

    df = df.dropna()
    #print(df)
    
    alfa = 1
    #L = 4 # not accounting for the vertical profile, for d=3m then L=3m
    n = 2
    #Ks = 1 #m/hr
    theta_s = 1
    theta_r = 0.2

    t_list = []
    x_axis = []
    y_axis = []
    for i in range(len(df)):
        q = df[q_column].iloc[i]
        k = df[k_column].iloc[i]
        d = df[d_column].iloc[i]
        t = df[t_column].iloc[i]
        if k>=q and t!=0:
            #q = q/Ks
            #k = k/Ks
            #d = alfa*d
            #t = (alfa*Ks*t)/(theta_s-theta_r)
            #L = alfa*d
            
            x = alfa*d
            #x = (k*t/(d))
            #y = ((alfa*d*d)/((t)))**(1-1/n) #best performance
            #y = (k*t/(d))**(1-1/n)
            y = (alfa*k*t)**(1-1/n) # best dimensionless
            #y_pred = (3.4743488481411937)*(x**(-0.43044273123818805))
            #t_pred = (y_pred*y_pred*alfa*d*d)/(Ks)
            #t_list.append(abs(t_pred-t))
            y_axis.append(y)
            x_axis.append(x)

    params, covariance = curve_fit(powerlaw_fit, x_axis, y_axis)
    # Extract the fitted parameters
    a_fit, b_fit = params

    # Print the results
    print(f"Fitted a: {a_fit} and b:{b_fit}")

    # Generate data points for the fitted curve
    x_fit = [min(x_axis), max(x_axis)]
    y_fit = [powerlaw_fit(x, a_fit, b_fit) for x in x_axis]
    R_square = r2_score(y_axis, y_fit)
    print(f"R2 = {R_square}")
    MSE = np.square(np.subtract(y_axis,y_fit)).mean()
    print(f"MSE = {MSE*10**-9} x10⁹")
    RMSE = math.sqrt(MSE)
    print(f"RMSE = {RMSE}")
    MAE = mean_absolute_error(y_axis, y_fit)
    print(f"MAE = {MAE}")
    #print(f"max t = {np.max(t_list)} min t = {np.min(t_list)}")
    
    y_fit = [powerlaw_fit(x, a_fit, b_fit) for x in x_fit]

    plt.figure(figsize=(16,9))
    plt.grid(True)
    plt.scatter(x_axis, y_axis, label="data points", s=30, facecolors='r', edgecolors='r')
    plt.plot(x_fit, y_fit, color="black", label="fitted line")
    #plt.scatter(x_axis, y_fit, s=30, facecolors='b', edgecolors='b')# color="black", label="fitted")
    #plt.annotate(f"R²={round(R_square, 3)}\nf(x)={round(a_fit, 3)}x^({round(b_fit, 3)})", xy=(0.0001, 1), color="black")
    plt.annotate(f"R²={round(R_square, 3)}\nf(x)={round(a_fit, 3)}x^({round(b_fit, 3)})", xy=(1, 7), color="black")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("αd (-)")
    plt.ylabel("(αkt)^(1-1/n) (-)")
    plt.legend()
    plt.show()

def plot_ss_profiles():
    plt.rcParams.update({'font.size': 22})
    folderpath='/p/project/cslts/miaari1/parflow_simulations/ss_profilesplot'
    z = list(range(41, 1, -1))
    z = [x*0.1 for x in z]

    files = [os.path.join(folderpath, x) for x in os.listdir(folderpath)]
    for file in files:
        name = os.path.basename(file)
        name = name.replace(".pfb", "")
        namelist = name.split("_")
        q = namelist[0].replace("-",".")
        k = namelist[1].replace("-",".")
        t = namelist[2]
        qk = float(q)/float(k)

        data = SLOTH.sloth.IO.read_pfb(file)

        plt.plot(data[:,0,0],z, color="black")
        plt.annotate(f"q/k={qk}", xy=(data[:,0,0][-1], z[-1]), color="black")
    
    plt.gca().invert_yaxis()
    plt.xlim([-4, 0.2])
    plt.xlabel("Pressure (m)")
    plt.ylabel("Soil depth (m)")
    plt.show()

def check_nc():
    filepath = "/p/scratch/cslts/miaari1/cordex_data/training_data_regional.nc"
    ncfile = nc.Dataset(filepath)
    print(ncfile)
    variables = ncfile.variables
    print(variables)

def download_files():
    import requests
    import os
    filespath = "/p/scratch/cslts/miaari1/cordex_data/raw_data/"
    dates = [x for x in range(1998, 2018)]
    for date in dates:
        print(f"downloading for {date}")
        url = f'https://datapub.fz-juelich.de/slts/cordex/data/pgw/pgw_EUR-11_ECMWF-ERAINT_evaluation_r1i1p1_FZJ-IBG3-TERRSYSMP11_v1_day_{date}0101-{date}1231.nc'
        data = requests.get(url).content
        filename = f"pgw_EUR-11_ECMWF-ERAINT_evaluation_r1i1p1_FZJ-IBG3-TERRSYSMP11_v1_day_{date}0101-{date}1231.nc"
        with open(os.path.join(filespath, f"{filename}"), "wb") as f:
                f.write(data)
        f.close()

def plot_analytical_numerical():
    # read from numerical solution
    numerical_path = "/p/project/cslts/miaari1/python_scripts/outputs/numerical_pressure.csv"
    numerical_solution = pd.read_csv(numerical_path)
    y_axis = numerical_solution["z"].to_list()
    x_axis10 = numerical_solution["time=10"].to_list()
    x_axis100 = numerical_solution["time=100"].to_list()
    x_axis1091 = numerical_solution["time=1091"].to_list()
    plt.plot(x_axis10, y_axis, color="red")
    plt.annotate(f"t=10", xy=(x_axis10[-1], y_axis[-1]), color="black")
    plt.plot(x_axis100, y_axis, color="red")
    plt.annotate(f"t=100", xy=(x_axis100[-1], y_axis[-1]), color="black")
    plt.plot(x_axis1091, y_axis, color="red")
    plt.annotate(f"t=1091", xy=(x_axis1091[-1], y_axis[-1]), color="black")
    
    

    # read from analytical solution
    analytical_path = "/p/project/cslts/miaari1/python_scripts/outputs/analytical_pressure_modalfa.csv"
    analytical_solution = pd.read_csv(analytical_path)
    y_axis = analytical_solution["z"].to_list()
    x_axis10 = analytical_solution["time=10"].to_list()
    x_axis100 = analytical_solution["time=100"].to_list()
    x_axis1091 = analytical_solution["time=1091"].to_list()
    plt.plot(x_axis10, y_axis, color="blue")
    plt.annotate(f"t=10", xy=(x_axis10[-2], y_axis[-2]), color="black")
    plt.plot(x_axis100, y_axis, color="blue")
    plt.annotate(f"t=100", xy=(x_axis100[-2], y_axis[-2]), color="black")
    plt.plot(x_axis1091, y_axis, color="blue")
    plt.annotate(f"t=1091", xy=(x_axis1091[-2], y_axis[-2]), color="black")

    #plt.gca().invert_yaxis()
    plt.xlim([-4, 0.2])
    plt.xlabel("Pressure (m)")
    plt.ylabel("Soil depth (m)")
    plt.show()


def analytical_pressure():
    import math
    # define variables in cm and hr
    L_star = 350 #cm
    #L_star = 400 #cm
    Ks = 10 #cm/hr
    #alfa = 0.1 # 1/cm
    # alfa = 0.0262 perfect for t=1091
    alfa = 0.025616931634881337 # NOT 1/cm but fitted based on K from Van Genuchten
    #Ss = 0.4
    Ss = 1
    #Sr = 0.06
    Sr = 0.2
    #qa_star = 0.1 #cm/hr
    qa_star = 0 #cm/hr
    qb_star = 0.01 #cm/hr
    #qb_star = 0.1 #cm/hr
    pressure_0 = 0 #cm

    # define dimensionless parameters
    L = alfa*L_star
    qa = qa_star/Ks
    qb = qb_star/Ks

    # get the roots for this L
    roots = find_roots_from_plot(L)

    # define layers
    data = {}
    data = {"z": [*range(1, L_star+1, 1)]}
    # calculate for timesteps
    timesteps = range(10, 2002, 1)
    #timesteps = [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    #timesteps = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]
    #timesteps = [1090,1091]
    for t_star in timesteps:
        print(t_star)
        data[f"time={t_star}"] = []
        t = (alfa*Ks*t_star)/(Ss-Sr)
        for z_star in data["z"]:
            #plotit = True
            
            # for z_star=0 bottom layer, z_star=L_star top layer
            z = alfa*z_star
            #if z>0:
            #    plotit = False
            residues = sum_residue(L, t, z, roots)
            #if not plotit:
            #    print(residues)
            #    return
            #print(f"for t_star={t_star} and z={z_star} the residue is: {residues}")
            
            K = qb - (qb - np.exp(alfa*pressure_0))*np.exp(-z) - 4*(qb - qa)*(np.exp((L-z)/2)*np.exp(-t/4))*residues

            K_star = K*Ks
            pressure = (np.log(K_star/Ks))/alfa
            data[f"time={t_star}"].append(pressure)
        #diff = []
        #for i in range(len(data["z"])):
        #    if t_star-1>=timesteps[0] and abs(abs(data[f"time={t_star-1}"][i]) - abs(data[f"time={t_star}"][i])) < 1e-7:
        #        diff.append(True)
        #if len(diff) == len(data["z"]):
        #    print(f"timestep= {t_star}")
        #    return


        if  t_star-1>=timesteps[0] and np.square(np.subtract(data[f"time={t_star-1}"],data[f"time={t_star}"])).mean() < 1e-10:
            print("steady state reached on MSE")
            print(f"timestep= {t_star}")
            print(f'MSE= {np.square(np.subtract(data[f"time={t_star-1}"],data[f"time={t_star}"])).mean()}')
            return
            
    
    df = pd.DataFrame(data)
    print(df)
    df = df/100
    #t1 = df["time=1090"].to_list()
    #t2 = df["time=1091"].to_list()

    #MSE = np.square(np.subtract(t1,t2)).mean()
    #print(f"MSE = {MSE}")
    #RMSE = math.sqrt(MSE)
    #print(f"RMSE = {RMSE}")
    

def parflow_namelist():
    import random
    dir_path = "/p/scratch/cslts/miaari1/testcases"
    filepath = "/p/project/cslts/miaari1/python_scripts/infiltration.tcl"
    VG_params_path = "/p/project/cslts/miaari1/python_scripts/WaterFlowParameters.csv"
    f = open(filepath, "r")
    namelist = f.readlines()
    f.close()

    k_range = [0.00001, 10]
    q_range = [0.00001, 10]
    #alfa_range = [0.5, 10]
    #n_range = [1, 3]
    d_range = [1, 10]
    #thetar_range = [0, 0.2]
    #thetas_range = [0.5, 1]
    VG_params = pd.read_csv(VG_params_path)
    VG_indexes = [0,1,2,3,4,5,6,7,8,9,10,11]

    k_namelist = namelist[64]
    q_namelist = namelist[214]
    d_namelist = namelist[54]
    Nz_namelist = namelist[31]
    alfa_namelist1 = namelist[142]
    alfa_namelist2 = namelist[151]
    n_namelist1 = namelist[143]
    n_namelist2 = namelist[152]
    thetar_namelist = namelist[153]
    thetas_namelist = namelist[154]

    for i in range(12):
        new_namelist = namelist
        case_path = os.path.join(dir_path, f"test_case{i}")
        os.mkdir(case_path)

        #VG_index = random.choice(VG_indexes)
        VG_index = i
        alfa = VG_params["Alpha"].iloc[VG_index]
        n = VG_params["n"].iloc[VG_index]
        theta_r = VG_params["Sr"].iloc[VG_index]
        theta_s = VG_params["Ss"].iloc[VG_index]
        k = VG_params["Ks"].iloc[VG_index]

        #q_k = random.uniform(0.0001, 0.001)
        q = 0.001
        #q = q_k * k
        #k = random.uniform(k_range[0], k_range[1])
        #q = random.uniform(q_range[0], min(k, 1))
        
        new_namelist[64] = k_namelist.replace("0.01",f"{k}")
        new_namelist[214] = q_namelist.replace("0.001", f"{q}")
        #alfa = round(random.uniform(alfa_range[0], alfa_range[1]),2)
        #n = round(random.uniform(n_range[0], n_range[1]),2)
        new_namelist[142] = alfa_namelist1.replace("1.0", f"{alfa}")
        new_namelist[151] = alfa_namelist2.replace("1.", f"{alfa}")
        new_namelist[143] = n_namelist1.replace("2.0", f"{n}")
        new_namelist[152] = n_namelist2.replace("2.0", f"{n}")
        #theta_r = round(random.uniform(thetar_range[0], thetar_range[1]),2)
        #theta_s = round(random.uniform(thetas_range[0], thetas_range[1]),2)
        new_namelist[153] = thetar_namelist.replace("0.2", f"{theta_r}")
        new_namelist[154] = thetas_namelist.replace("1.0", f"{theta_s}")
        #d = round(random.uniform(d_range[0], d_range[1]),1)
        d = 4.0
        new_namelist[54] = d_namelist.replace("4.0", f"{d}")
        new_namelist[31] = Nz_namelist.replace("40", f"{int(d*10)}")

        settings = ["k,q,alfa,n,theta_r,theta_s,d",f"{k}",f"{q}",f"{alfa}",f"{n}",f"{theta_r}", f"{theta_s}",f"{d}"]
        with open(os.path.join(dir_path, f"test_case{i}", "settings.txt"),'w') as settingsfile:
            settingsfile.write('\n'.join(settings))
        settingsfile.close()

        with open(os.path.join(dir_path, f"test_case{i}", "infiltration.tcl"), 'w') as tclfile:
            tclfile.write('\n'.join(new_namelist))
        tclfile.close()
    return

def simulation_bash():
    dir_path = "/p/scratch/cslts/miaari1/testcases"
    filepath = "/p/project/cslts/miaari1/submit_simulation.sh"
    f = open(filepath, "r")
    sbatch = f.readlines()
    f.close()
    sbatch = sbatch[:13]
    for i in range(12):
        #case_path = os.path.join(dir_path, f"test_case{i}")
        case_path = f"cd /p/scratch/cslts/miaari1/testcases/test_case{i}"
        run_command = "tclsh infiltration.tcl"
        sbatch.append(case_path)
        sbatch.append(run_command)
    
    with open("/p/project/cslts/miaari1/submit_simulation_advanced.sh", 'w') as sbatchfile:
        sbatchfile.write('\n'.join(sbatch))
    sbatchfile.close()


def steadystate_kinsol():
    dir_path = "/p/scratch/cslts/miaari1/testcases"
    data = {"q": [], "k": [], "d": [], "alfa": [], "n": [], "theta_r": [], "theta_s": [], "time": []}
    for case_index in range(12):
        reached = False
        case_dir = os.path.join(dir_path, f"test_case{case_index}")
        f = open(os.path.join(case_dir, "settings.txt"), "r")
        settings = f.readlines()
        f.close()

        

        kinsolfile = os.path.join(case_dir, "infiltration.out.kinsol.log")
        filedata = open(kinsolfile)
        lines = filedata.readlines()
        time_kinsol = [x for x in lines if "KINSOL starting step for time" in x or "KINSolInit nni=    0  fnorm=" in x]
        for i in range(1, len(time_kinsol)):
            if "KINSolInit nni=    0  fnorm=" not in time_kinsol[i] and "KINSolInit nni=    0  fnorm=" not in time_kinsol[i-1]:
                index = i-1
                time = re.findall("\d+\.\d+",time_kinsol[index])
                #print(time)
                time = float(time[0])
                data["time"].append(time)
                data["k"].append(float(settings[1].replace("\n","")))
                data["q"].append(float(settings[2].replace("\n","")))
                data["alfa"].append(float(settings[3].replace("\n","")))
                data["n"].append(float(settings[4].replace("\n","")))
                data["theta_r"].append(float(settings[5].replace("\n","")))
                data["theta_s"].append(float(settings[6].replace("\n","")))
                data["d"].append(float(settings[7].replace("\n","")))
                reached = True
                break
            if i == len(time_kinsol)-1 and not reached:
                print(f"not reached: {case_index}")
    print(len(data["time"]))
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs", "testcases_constantq_constantd_v3.csv"), index=False)


def append_csv():
    df_path = "/p/project/cslts/miaari1/python_scripts/outputs/testcases_appended.csv"
    df1_path = "/p/project/cslts/miaari1/python_scripts/testcases_4.csv"
    df = pd.read_csv(df_path)
    df1 = pd.read_csv(df1_path)

    df = pd.concat([df, df1], ignore_index=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv('/p/project/cslts/miaari1/python_scripts/outputs/testcases_appended.csv', index=False)


def plot_pdf(data):
    import numpy as np
    import matplotlib.pyplot as plt
    
    #data = np.random.normal(0, 1, 1000)  # Example data

    counts, bin_edges = np.histogram(data, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, counts
    plt.plot(bin_centers, counts, linestyle='-', marker='o')
    plt.xlabel('X-Axis Label')
    plt.ylabel('PDF')
    plt.title('Probability Density Function')
    plt.grid(True)
    plt.show()

def optimize_power(initial_data):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # Generate initial data (you should replace this with your actual data)
    #initial_data = np.random.exponential(scale=2, size=1000)

    # Define the power-law function
    def power_law(x, c):
        return x**c

    # Define an initial guess for the parameters
    initial_guess = [-2]

    # Fit the data to the power-law function
    params, covariance = curve_fit(power_law, initial_data, np.arange(len(initial_data)), p0=initial_guess)

    # Extract the optimized parameters
    optimized_c = params
    print(optimized_c)
    # Apply the optimized constant power    
    transformed_data = initial_data ** optimized_c

    # Plot initial and transformed data
    plt.hist(initial_data, bins=30, alpha=0.5, label='Initial Data')
    plt.hist(transformed_data, bins=30, alpha=0.5, label=f'Transformed Data (Power={optimized_c:.2f})')

    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Initial and Transformed Data')
    plt.show()

def plot_ss_profiles():
    plt.rcParams.update({'font.size': 22})
    folderpath='/p/project/cslts/miaari1/python_scripts/parflow'
    z = [4-z/100 for z in range(5,400, 10)]

    data_loam = SLOTH.sloth.IO.read_pfb(os.path.join(folderpath, "loamsoil", "infiltration.out.press.00419.pfb"))
    data_clay = SLOTH.sloth.IO.read_pfb(os.path.join(folderpath, "claysoil", "infiltration.out.press.00237.pfb"))
    data_loam_qk = SLOTH.sloth.IO.read_pfb(os.path.join(folderpath, "loamsoil_qk", "infiltration.out.press.02813.pfb"))
    data_clay_qk = SLOTH.sloth.IO.read_pfb(os.path.join(folderpath, "claysoil_qk", "infiltration.out.press.02851.pfb"))
    data_sandy_qk = SLOTH.sloth.IO.read_pfb(os.path.join(folderpath, "sandysoil_qk", "infiltration.out.press.00121.pfb"))
    data_sandy = SLOTH.sloth.IO.read_pfb(os.path.join(folderpath, "sandysoil", "infiltration.out.press.00558.pfb"))
    plt.plot(data_loam[:,0,0],z, label="Loam soil")
    plt.plot(data_clay[:,0,0],z, label="Clay soil")
    plt.plot(data_loam_qk[:,0,0],z, label="Loam soil qk")
    plt.plot(data_clay_qk[:,0,0],z, label="Clay soil qk")
    plt.plot(data_sandy_qk[:,0,0],z, label="Sandy soil qk")
    plt.plot(data_sandy[:,0,0],z, label="Sandy soil")
    
    plt.gca().invert_yaxis()
    plt.xlim([-4, 0.0])
    plt.xlabel("Pressure (m)")
    plt.ylabel("Soil depth (m)")
    plt.legend()
    plt.show()

#plot_ss_profiles()
plot_qk_ss()
#plot_qk_ss_drainage()
#parflow_namelist()
#simulation_bash()
#steadystate_kinsol()
#append_csv()