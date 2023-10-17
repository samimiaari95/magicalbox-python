import tensorflow as tf
from tensorflow import keras
import numpy as np
import xarray as xr

def normalise_data(data, mu=None, std=None, dims=None):
        '''
        Normalise the data in an xarray dataset
        '''
        
        if mu is None and std is None:
            if not dims:
                dims = list(data.dims)
            
            mu = data.mean(dim=dims)
            std = data.std(dim=dims)
            
        data_out = (data-mu)/std
        
        return data_out, mu, std

def pre_process_data():
    # import packages
    import matplotlib.pyplot as plt
    #%matplotlib inline
    import os
    import sys
    #import wandb
    #from wandb.integration.keras import WandbCallback  

    import xarray as xr
    import datetime as dt
    import numpy as np
    import scipy
    import cftime
    # %load ./solutions/task_2-1_sol.py
    # Task 2.1:

    # set the base data directory:
    #data_dir = "/p/project/training2326/hickman2/hpsc-terrsys-2023/01_Large_Data/Exercise/data/processed"
    home_dir = "/p/project/cslts/miaari1/cordex_data/postprocessed/"
    data_dir = "/p/scratch/cslts/miaari1/cordex_data"
    # set the path to specific training, validation and test datasets:
    ftrain_nc = os.path.join(data_dir, "training_data_regional.nc")
    fval_nc = os.path.join(data_dir, "val_data_regional.nc")
    ftest_nc = os.path.join(data_dir, "test_data_regional.nc")

    # optionally check the datapath:
    print(ftrain_nc)

    # read the netcdf data using xarray
    # `xarray.open_dataset`
    ds_train = xr.open_dataset(ftrain_nc)
    ds_val = xr.open_dataset(fval_nc) 
    ds_test = xr.open_dataset(ftest_nc)
    # Select temperature from the xarray DataSet to produce a DataArray
    ds_train_tas = ds_train.pgw
    ds_val_tas = ds_val.pgw
    ds_test_tas = ds_test.pgw
    #print(ds_val_tas)
    #ds_train_tas.isel(time=3).plot(vmin=250, vmax=300, figsize=(12, 5))

    #ds_val_tas.isel(time=3).plot(vmin=250, vmax=300, figsize=(12, 5))

    #ds_test_tas.isel(time=3).plot(vmin=250, vmax=300, figsize=(12, 5))
    #plt.title("A better title?")
    #plt.show()

    # As an aside: this is how we could select a particular time from a dataset - if we had hourly data, for example.

    ds_train_tas = ds_train_tas.sel(time=ds_train_tas.time.dt.hour == 10)
    ds_val_tas = ds_val_tas.sel(time=ds_val_tas.time.dt.hour == 10)
    ds_test_tas = ds_test_tas.sel(time=ds_test_tas.time.dt.hour == 10)

    # select top layer pressure
    ds_train_tas = ds_train_tas.sel(lev=ds_train_tas.lev == 15.0)
    ds_val_tas = ds_val_tas.sel(lev=ds_val_tas.lev == 15.0)
    ds_test_tas = ds_test_tas.sel(lev=ds_test_tas.lev == 15.0)

    # remove layers dimension from the variables
    ds_train_tas = ds_train_tas.squeeze("lev")
    ds_val_tas = ds_val_tas.squeeze("lev")
    ds_test_tas = ds_test_tas.squeeze("lev")

    # coarsen our fine resolution data
    ds_train_tas_coarse = ds_train_tas.coarsen(rlon=2, rlat=2, boundary='exact').mean()
    ds_val_tas_coarse = ds_val_tas.coarsen(rlon=2, rlat=2, boundary='exact').mean()
    ds_test_tas_coarse = ds_test_tas.coarsen(rlon=2, rlat=2, boundary='exact').mean()
    print(ds_train_tas_coarse.shape)
    # interpolate back to the fine resolution (the new matrix will have different values than the original one)
    ds_train_tas_coarse = ds_train_tas_coarse.interp_like(ds_train_tas, method="nearest")
    ds_val_tas_coarse = ds_val_tas_coarse.interp_like(ds_val_tas, method="nearest")
    ds_test_tas_coarse = ds_test_tas_coarse.interp_like(ds_test_tas, method="nearest")

    # set nans to the mean value - not perfect!
    ds_train_tas = ds_train_tas.fillna(ds_train_tas.mean())
    ds_train_tas_coarse = ds_train_tas_coarse.fillna(ds_train_tas.mean())

    ds_val_tas = ds_val_tas.fillna(ds_train_tas.mean())
    ds_val_tas_coarse = ds_val_tas_coarse.fillna(ds_train_tas.mean())

    ds_test_tas = ds_test_tas.fillna(ds_train_tas.mean())
    ds_test_tas_coarse = ds_test_tas_coarse.fillna(ds_train_tas.mean())

    # Normalise the training data
    t2m_in_train, t2m_in_train_mu, t2m_in_train_std = normalise_data(ds_train_tas)
    t2m_tar_train, t2m_tar_train_mu, t2m_tar_train_std = normalise_data(ds_train_tas_coarse)

    # Normalise the validation data
    t2m_in_val, _, _ = normalise_data(ds_val_tas, mu=t2m_in_train_mu, std=t2m_in_train_std) #normalize t2m_in variable
    t2m_tar_val, _, _ = normalise_data(ds_val_tas_coarse, mu=t2m_tar_train_mu, std=t2m_tar_train_std) #normalize t2m_tar variable

    # Task: normalise the testing data:
    t2m_in_test, _, _ = normalise_data(ds_test_tas, mu=t2m_in_train_mu, std=t2m_in_train_std)
    t2m_tar_test, _, _ = normalise_data(ds_test_tas_coarse, mu=t2m_in_train_mu, std=t2m_in_train_std)

    # %load ./solutions/task_2-15_sol.py
    #t2m_in_test, t2m_in_test_mu, t2m_in_test_std = normalise_data(ds_test_tas, mu=t2m_in_train_mu, std=t2m_in_train_std) #normalize t2m_in variable
    #t2m_tar_test, t2m_tar_test_mu, t2m_tar_test_std = normalise_data(ds_test_tas_coarse, mu=t2m_tar_train_mu, std=t2m_tar_train_std) #normalize t2m_tar variable

    # How different are the mean and std values of the unnormalised and normalised validation and test dataset?
    # Validation, unnormalised:
    print(ds_val_tas.mean(), ds_val_tas.std())

    # Validation, normalised:
    print(t2m_in_val.mean(), t2m_in_val.std())

    unnorm_norm_mean = ds_val_tas.mean()/t2m_in_val.mean()
    print("the ratio is")
    print(unnorm_norm_mean)
    print(t2m_in_train.mean())
    print(t2m_in_val.mean())
    print("##############second part###################")

    # Could include other variables here, hence we use concat in case we wanted to include another variable.
    input_train_data = xr.concat([t2m_in_train], dim="variable")
    target_train_data = xr.concat([t2m_tar_train], dim="variable")
    # re-order data as needed by the network
    input_train_data  = input_train_data.transpose("time", "rlat", "rlon", "variable")
    target_train_data = target_train_data.transpose("time", "rlat", "rlon", "variable")

    # Validation data
    input_val_data = xr.concat([t2m_in_val], dim="variable")
    target_val_data = xr.concat([t2m_tar_val], dim="variable")
    input_val_data  = input_val_data.transpose("time", "rlat", "rlon", "variable")
    target_val_data = target_val_data.transpose("time", "rlat", "rlon", "variable")

    # Test data
    input_test_data = xr.concat([t2m_in_test], dim="variable")
    target_test_data = xr.concat([t2m_tar_test], dim="variable")
    input_test_data  = input_test_data.transpose("time", "rlat", "rlon", "variable")
    target_test_data = target_test_data.transpose("time", "rlat", "rlon", "variable")

    # Finally, let's save this data - note, you may want to create a data directory.
    # training data
    #input_train_data.to_netcdf(os.path.join(home_dir, "input_train_data.nc"))

    input_train_data.to_netcdf("/p/scratch/cslts/miaari1/cordex_data/postprocessed/input_train_data.nc")
    target_train_data.to_netcdf("/p/scratch/cslts/miaari1/cordex_data/postprocessed/target_train_data.nc")

    # validation data
    input_val_data.to_netcdf("/p/scratch/cslts/miaari1/cordex_data/postprocessed/input_val_data.nc")
    target_val_data.to_netcdf("/p/scratch/cslts/miaari1/cordex_data/postprocessed/target_val_data.nc")

    # testing data
    input_test_data.to_netcdf("/p/scratch/cslts/miaari1/cordex_data/postprocessed/input_test_data.nc")
    target_test_data.to_netcdf("/p/scratch/cslts/miaari1/cordex_data/postprocessed/target_test_data.nc")

#def FFANN_example():
    # Generate synthetic data for classification
    # training data
    input_train_data = xr.open_dataarray("/p/scratch/cslts/miaari1/cordex_data/postprocessed/input_train_data.nc")
    target_train_data = xr.open_dataarray("/p/scratch/cslts/miaari1/cordex_data/postprocessed/target_train_data.nc")

    # validation data
    input_val_data = xr.open_dataarray("/p/scratch/cslts/miaari1/cordex_data/postprocessed/input_val_data.nc")
    target_val_data = xr.open_dataarray("/p/scratch/cslts/miaari1/cordex_data/postprocessed/target_val_data.nc")

    # testing data
    input_test_data = xr.open_dataarray("/p/scratch/cslts/miaari1/cordex_data/postprocessed/input_test_data.nc")
    target_test_data = xr.open_dataarray("/p/scratch/cslts/miaari1/cordex_data/postprocessed/target_test_data.nc")

    np.random.seed(0)
    #X = np.random.rand(100, 2)  # 100 samples with 2 features
    X_train = input_train_data.isel(variable=0).values
    y_train = target_train_data.isel(variable=0).values
    #y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary classification task
    # Split the data into training and testing sets
    #X_train, X_test = X[:80], X[80:]
    #y_train, y_test = y[:80], y[80:]
    X_test = input_val_data.isel(variable=0).values
    y_test = target_val_data.isel(variable=0).values
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(input_train_data.isel(variable=0).values.shape)

    # Create a simple feedforward neural network model
    # TODO check the shape provided in the downscaling unet, probably it should be the same here
    shape_in = (48, 64, 1)
    model = keras.Sequential([
        keras.layers.Dense(8, activation='sigmoid', input_shape=shape_in)#,
        #keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=256, verbose=2)

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(test_loss)
    print(test_acc)
    #model.save_weights('/p/project/cslts/miaari1/python_scripts/checkpoints/FFANN_weights')

    model = keras.Sequential([
        keras.layers.Dense(8, activation='sigmoid', input_shape=shape_in),
        #keras.layers.Dense(1024, activation='sigmoid')
    ])

    #model.load_weights('/p/project/cslts/miaari1/python_scripts/outputs/FFANN_weights')
    y_pred_test = model.predict(target_test_data.values, verbose=1)
    y_pred_trans = y_pred_test*t2m_in_train_std.squeeze().values + t2m_in_train_mu.squeeze().values
    y_pred_trans = xr.DataArray(y_pred_trans, coords=ds_test_tas.coords, dims=input_test_data.dims)


    t2m_tar = ds_test_tas.isel(time=3)
    y_pred = y_pred_trans.isel(time=3)
    diff_pred_tar =  y_pred - t2m_tar 
    t2m_tar.plot(vmin=-10, vmax=40, figsize=(12, 5))
    plt.savefig("/p/scratch/cslts/miaari1/cordex_data/postprocessed/t2m_tar.png")
    y_pred.plot(vmin=-10, vmax=40, figsize=(12, 5))
    plt.savefig("/p/scratch/cslts/miaari1/cordex_data/postprocessed/y_pred.png")
    diff_pred_tar.plot(vmin=-10, vmax=10, figsize=(12, 5))
    plt.title('Difference between prediction and original')
    plt.savefig("/p/scratch/cslts/miaari1/cordex_data/postprocessed/diff_org_pred.png")
    return

pre_process_data()