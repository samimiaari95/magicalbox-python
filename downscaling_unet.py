
def preprocess_rawdata():
    import os
    import random
    import glob
    import xarray as xr

    # First, let's define some functions.

    def load_dataarray_datetime(filename, variable):
        '''
        Parameters
        ----------
        filename : str
            Path to the netCDF file.
        variable : str
            Variable name to read from the netCDF file.

        Returns
        -------
        dataarray : xarray.DataArray
            DataArray of the variable.
        '''
        dataset = xr.open_dataset(filename)
        dataarray = dataset[variable]#[:][-1][:][:]
        #dataarray['time'] = dataarray.indexes['time'].to_datetimeindex()
        return dataarray

    def sample_days_dataarray(dataarray, no_of_samples=None):
        '''
        Parameters
        ----------
        dataarray : xarray.DataArray
            DataArray of the variable.
        no_of_samples : int
            Number of samples we wish to take from a particular year.

        Returns
        -------
        dataarray_random_days : xarray.DataArray
            DataArray with random days selected.
        '''    
        random_days = random.sample(range(0, len(dataarray.time)), no_of_samples)
        dataarray_random_days = dataarray.isel(time=random_days)
        return dataarray_random_days
        
    def select_region_dataarray(dataarray, rlat_slice=slice(160, 208), rlon_slice=slice(140,204)): 
        '''
        Parameters
        ----------
        dataarray : xarray.DataArray
            DataArray of the variable.
        rlat_slice : slice
            latitude range we wish to return
        rlon_slice : slice
            longitude range we wish to return

        Returns
        -------
        dataarray_region : xarray.DataArray
            DataArray over the specified region
        '''        
        dataarray_region = dataarray.isel(rlat=rlat_slice, rlon=rlon_slice)
        return dataarray_region


    # Now let's process the data!
    # Firstly set the directory where the raw .nc files are (change the filepaths!):

    #dir = '/p/project/training2326/hickman2/hpsc-terrsys-2023/01_Large_Data/Exercise/data/raw'
    rawdata_dir = "/p/scratch/cslts/miaari1/cordex_data/raw_data"
    #os.chdir(dir)

    # create a list of files in the directory
    files = glob.glob('/p/scratch/cslts/miaari1/cordex_data/raw_data/*.nc')

    # select 2 random files from the list to be the validation and testing data
    random_files = random.sample(files, 4)

    # remove these random files from the list for the validation and test datasets
    files.remove(random_files[0])
    files.remove(random_files[1])
    files.remove(random_files[2])
    files.remove(random_files[3])


    # take the validation and test data as full years of data
    ds_val_full1 = load_dataarray_datetime(random_files[0], "pgw")
    ds_val_full2 = load_dataarray_datetime(random_files[1], "pgw")
    ds_test_full1 = load_dataarray_datetime(random_files[2], "pgw")
    ds_test_full2 = load_dataarray_datetime(random_files[3], "pgw")

    # select region using defaults (see the function)
    ds_val1 = select_region_dataarray(ds_val_full1)
    ds_val2 = select_region_dataarray(ds_val_full2)
    ds_test1 = select_region_dataarray(ds_test_full1)
    ds_test2 = select_region_dataarray(ds_test_full2)

    # concatenate test and validation datasets
    ds_val = xr.concat([ds_val1, ds_val2], dim="time")
    ds_test = xr.concat([ds_test1, ds_test2], dim="time")

    # Now dealing with the training data - we want to sample a certain number of days from each year, and select the same region.

    # create an empty list
    result_list = []
    # loop through the remaining files in the directory
    for file in files:
        # load netcdf and select groundwater pressure pgw(time, lev, rlat, rlon)
        dataarray = load_dataarray_datetime(file, "pgw")
        ## select 30 samples from the particular year
        #sample_days_da = sample_days_dataarray(dataarray, no_of_samples=30)
        # select the region using the defaults in the function
        sample_days_da_region = select_region_dataarray(dataarray)
        # append the data to an empty xarray dataset
        result_list.append(sample_days_da_region)


    ds_train = xr.concat(result_list, dim="time")


    # write all these to file - but change the filepath!!
                                                #BEWARE#
    ds_train.to_netcdf('/p/scratch/cslts/miaari1/cordex_data/training_data_regional.nc')
    ds_val.to_netcdf('/p/scratch/cslts/miaari1/cordex_data/val_data_regional.nc')
    ds_test.to_netcdf('/p/scratch/cslts/miaari1/cordex_data/test_data_regional.nc')

def downscaling_ml():
    # Function to normalise the data
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
    def conv_block(inputs, num_filters: int, kernel: tuple = (3,3), padding: str = "same",
               activation: str = "relu", kernel_init: str = "he_normal", l_batch_normalization: bool = True):
        """
        A convolutional layer with optional batch normalization
        :param inputs: the input data with dimensions nx, ny and nc
        :param num_filters: number of filters (output channel dimension)
        :param kernel: tuple indictating kernel size
        :param padding: technique for padding (e.g. "same" or "valid")
        :param activation: activation fuction for neurons (e.g. "relu")
        :param kernel_init: initialization technique (e.g. "he_normal" or "glorot_uniform")
        """
        
        x = Conv2D(num_filters, kernel, padding=padding, kernel_initializer=kernel_init)(inputs)
        if l_batch_normalization:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    def conv_block_n(inputs, num_filters, n=2, kernel=(3,3), padding="same", activation="relu", 
                     kernel_init="he_normal", l_batch_normalization=True):
        """
        Sequential application of two convolutional layers (using conv_block).
        """
        
        x = conv_block(inputs, num_filters, kernel, padding, activation,
                    kernel_init, l_batch_normalization)
        for i in np.arange(n-1):
            x = conv_block(x, num_filters, kernel, padding, activation,
                        kernel_init, l_batch_normalization)
        
        return x
    
    def encoder_block(inputs, num_filters, kernel_maxpool: tuple=(2,2), l_large: bool=True):
        """
        One complete encoder-block used in U-net
        """
        if l_large:
            x = conv_block_n(inputs, num_filters, n=2)
        else:
            x = conv_block(inputs, num_filters)
            
        p = MaxPool2D(kernel_maxpool)(x)
        
        return x, p

    # Fill all missing parts indicated by '...'.

    # Task 2.3: 
    def decoder_block(inputs, skip_features, num_filters, kernel: tuple=(3,3), strides_up: int=2, padding: str= "same", 
                    activation="relu", kernel_init="he_normal", l_batch_normalization: bool=True):
        """
        One complete decoder block used in U-net (reverting the encoder)
        """
        
        x = Conv2DTranspose(num_filters, (strides_up, strides_up), strides=strides_up, padding="same")(inputs)
        x = Concatenate()([x, skip_features])
        x = conv_block_n(x, num_filters, 2, kernel, padding, activation, kernel_init, l_batch_normalization)
    
        return x

    def build_unet(input_shape, channels_start=56, z_branch=False):
        
        inputs = Input(input_shape)
        
        """ encoder """
        s1, e1 = encoder_block(inputs, channels_start, l_large=True)
        s2, e2 = encoder_block(e1, channels_start*2, l_large=False)
        s3, e3 = encoder_block(e2, channels_start*4, l_large=False)
        
        """ bridge encoder <-> decoder """
        b1 = conv_block(e3, channels_start*8)
        
        """ decoder """
        d1 = decoder_block(b1, s3, channels_start*4) 
        #print(d1, s2)
        d2 = decoder_block(d1, s2, channels_start*2)
        d3 = decoder_block(d2, s1, channels_start)
        
        output_temp = Conv2D(1, (1,1), kernel_initializer="he_normal", name="output_temp")(d3)
        if z_branch:
            output_z = Conv2D(1, (1, 1), kernel_initializer="he_normal", name="output_z")(d3)

            model = Model(inputs, [output_temp, output_z], name="t2m_downscaling_unet_with_z")
        else:    
            model = Model(inputs, output_temp, name="t2m_downscaling_unet")
        
        return model
    
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



    ############################
    print("importing")
    # import tensorflow and required modules from Keras API
    import tensorflow as tf
    import tensorflow.keras.utils as ku
    # all the layers used for U-net
    from tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                                        Conv2DTranspose, Input, MaxPool2D)
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    print("module loaded")

    # Read in saved data if need be!

    # training data
    #input_train_data = xr.open_dataarray("data/input_train_data.nc")
    #target_train_data = xr.open_dataarray("data/target_train_data.nc")

    # validation data
    #input_val_data = xr.open_dataarray("data/input_val_data.nc")
    #target_val_data = xr.open_dataarray("data/target_val_data.nc")

    # testing data
    #input_test_data = xr.open_dataarray("data/input_test_data.nc")
    #target_test_data = xr.open_dataarray("data/target_test_data.nc")

    # fill nans with mean if required:

    #input_train_data = input_train_data.fillna(input_train_data.mean())
    #target_train_data = target_train_data.fillna(input_train_data.mean())

    #input_val_data = input_val_data.fillna(input_train_data.mean())
    #target_val_data = target_val_data.fillna(input_train_data.mean())

    #input_test_data = input_test_data.fillna(input_train_data.mean())
    #target_test_data = target_test_data.fillna(input_train_data.mean())

    ################## Build the architecture #############

    # Think here - what does the shape_in mean?
    print("building")
    shape_in = (48, 64, 1)
    unet_model = build_unet(shape_in, z_branch=False)
    print("module built")

    # ku.plot_model(unet_model, show_shapes=True)

    # Let's view and summarise this model:

    #unet_model.summary()


    #    If WandB is available in your environment, follow this. Otherwise, please skip to Compile and Train U-Net.
    #    Introducing Weights and Bias (wandb)

    #    There are increasing users in ML community using weights and biases, or similar tools. It is a very nice tool for machine learning experiment tracking, dataset versioning, and project collaboration. Think of it like an experiment log, or a lab book - but without actually having to write down all the experiments ourselves! In the following hands-on section, we will integrate wandb into TensorFlow. To start with wandb, firstly you need to sign up account:
    #    Sign up on wandb

    #    Sign up for a free account at https://wandb.ai/site and then login to your wandb account.
    #    Login to the wandb library on your machine. You will find your API key here: https://wandb.ai/authorize.
    # Initilize a new wandb run

    #wandb_dir = f"/p/project/training2326/hickman2/hpsc-terrsys-2023/{os.environ['USER']}"              

    #wandb.init(dir=wandb_dir)
    # overwrite the run name (like snowy-owl-10) with the run ID thata you can use this snippet
    #wandb.run.name = f"{os.environ['USER']}" + "test2"

    target_train_data.isel(variable=0)
    ### Default values for *hyper-parameters*, *if we use WandB*
    #config = wandb.config # Config is a variable that holds and saves hyperparameters and inputs
    #config.learning_rate = 5*10**(-2)
    #config.batch_size = 32
    #config.epochs = 30

    # Set our hyperparameters. What do these mean?

    learning_rate = 10**(-3)
    batch_size = 64
    epochs = 30
    unet_model.compile(optimizer=Adam(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError())

    # Here we only optimize the loss function on 2m-temperature as output, the U-Net is trained with 32 batch size and 10 epochs. 
    history = unet_model.fit(x=target_train_data.isel(variable=0).values, y=input_train_data.values, batch_size=batch_size, 
                            #callbacks = [WandbCallback(validation_data=(input_val_data.values, target_val_data.isel(variable=0).values),
                            #                          training_data=(input_train_data.values,target_train_data.isel(variable=0).values))],
                            epochs=epochs, validation_data=(target_val_data.isel(variable=0).values, input_val_data.values))
    # save the weights

    unet_model.save_weights('/p/project/cslts/miaari1/python_scripts/checkpoints/unet_weights')
    # We can load the weights again as follows:

    # Create a new model instance
    unet_model = build_unet(shape_in, z_branch=False)

    # Restore the weights
    unet_model.load_weights('/p/project/cslts/miaari1/python_scripts/outputs/unet_weights')
    
    # TODO continue from here
    # Remind us of this data

    # Prediction: generate the downscaled fields
    y_pred_test = unet_model.predict(target_test_data.values, verbose=1)
    # denormalise...
    y_pred_trans = y_pred_test*t2m_in_train_std.squeeze().values + t2m_in_train_mu.squeeze().values
    # and make it into an xarray DataArray 
    y_pred_trans = xr.DataArray(y_pred_trans, coords=ds_test_tas.coords, dims=input_test_data.dims)
    # Have a look at the predictions...

    #ds_test_tas.time[0:10]
    # choose the date you would like to plot here! Has to be in the test dataset.
    t2m_in = ds_test_tas_coarse.isel(time=3)
    #t2m_in, t2m_tar = ds_test_tas_coarse.isel(time=3), \
    #                ds_test_tas.isel(time=3)    # get the desired variables from the dataset
    t2m_tar = ds_test_tas.isel(time=3)
    y_pred = y_pred_trans.isel(time=3)
    print(y_pred)
    print(t2m_in.shape)

    # transform the unit from Kelvin to degree Celcius
    #t2m_in, t2m_tar, y_pred = t2m_in - 273.15, t2m_tar - 273.15, y_pred - 273.15           
    #calculate the differences between t2m_in and t2m_tar
    diff_in_tar = t2m_in - t2m_tar 

    #calculate the differences between y_pred and t2m_tar
    diff_pred_tar =  y_pred - t2m_tar 

    # Add some titles so we know which plot is which?

    t2m_in.plot(vmin=-10, vmax=40, figsize=(12, 5))
    plt.savefig("/p/scratch/cslts/miaari1/cordex_data/postprocessed/t2m_in.png")
    t2m_tar.plot(vmin=-10, vmax=40, figsize=(12, 5))
    plt.savefig("/p/scratch/cslts/miaari1/cordex_data/postprocessed/t2m_tar.png")
    y_pred.plot(vmin=-10, vmax=40, figsize=(12, 5))
    plt.savefig("/p/scratch/cslts/miaari1/cordex_data/postprocessed/y_pred.png")

    # What other plots could you make here?

    # Plot the differences?
    #Task 2.4-1

    diff_in_tar.plot(vmin=-10, vmax=10, figsize=(12, 5))
    plt.title('Difference between original and coarsened')
    plt.savefig("/p/scratch/cslts/miaari1/cordex_data/postprocessed/diff_org_coars.png")

    diff_pred_tar.plot(vmin=-10, vmax=10, figsize=(12, 5))
    plt.title('Difference between prediction and original')
    plt.savefig("/p/scratch/cslts/miaari1/cordex_data/postprocessed/diff_org_pred.png")

    # What about some other metrics?

    # RMSE
    mse_dom = ((ds_test_tas - y_pred_trans[...,0])**2).mean(dim=["time","rlat", "rlon"])
    rmse_dom = np.sqrt(mse_dom)
    print(f"global averaged RMSE of downscaled 2m temperature: {rmse_dom.values}")
    return print("successfully finished !!!")

    #wandb.init(dir=wandb_dir)
    # overwrite the run name (like snowy-owl-10) with the run ID thata you can use this snippet
    #wandb.run.name = {os.environ['USER']} + "test2" 

    ### Default values for hyper-parameters

    learning_rate = 5*10**(-2)
    batch_size = 32
    epochs = 30 # or more? Is more epochs always likely to be better?

    unet_model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mae")
    #Here we only optimize the loss function on 2m-temperature as output, the U-Net is trained with 32 batch size and 10 epochs. 
    history = unet_model.fit(x=input_train_data.values, y=target_train_data.isel(variable=0).values, batch_size=batch_size, 
                            #callbacks = [WandbCallback(validation_data=(input_val_data.values, target_val_data.isel(variable=0).values),
                            #                          training_data=(input_train_data.values,target_train_data.isel(variable=0).values))],
                            epochs=epochs, validation_data=(input_val_data.values, target_val_data.isel(variable=0).values))
    return


downscaling_ml()