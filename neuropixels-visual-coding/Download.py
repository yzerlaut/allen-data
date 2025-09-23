# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: allensdk
#     language: python
#     name: allensdk
# ---

# + [markdown] papermill={"duration": 0.007349, "end_time": "2023-11-30T06:23:22.921624", "exception": false, "start_time": "2023-11-30T06:23:22.914275", "status": "completed"} pycharm={"name": "#%% md\n"}
# # Accessing Neuropixels Visual Coding Data
#
# ## Tutorial overview
#
# This Jupyter notebook covers the various methods for accessing the Allen Institute Neuropixels Visual Coding dataset. We will go over how to request data, where it's stored, and what the various files contain.  If you're having trouble downloading the data, or you just want to know more about what's going on under the hood, this is a good place to start.
#
# Currently, we do not have a web interface for browsing through the available cells and experiments, as with the [two-photon imaging Visual Coding dataset](http://observatory.brain-map.org/visualcoding). Instead, the data must be retrieved through the AllenSDK (Python 3.6+), or via requests sent to [api.brain-map.org](http://mouse.brain-map.org/static/api).
#
# Functions related to data analysis will be covered in other tutorials. For a full list of available tutorials, see the [SDK documentation](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html).
#
# ## Options for data access
#
# The **`EcephysProjectCache` object of the AllenSDK** is the easiest way to interact with the data. This object abstracts away the details of on-disk file storage, and delivers the data to you as ready-to-analyze Python objects. The cache will automatically keep track of which files are stored locally, and will download additional files on an as-needed basis. Usually you won't need to worry about how these files are structured, but this tutorial will cover those details in case you want to analyze them without using the AllenSDK (e.g., in Matlab). This tutorial begins with <a href='#Using-the-AllenSDK-to-retrieve-data'>an introduction to this approach</a>.
#
# If you have an **Amazon Web Services (AWS)** account, you can use an `EcephysProjectCache` object to access the data via the Allen Brain Observatory Simple Storage Service (S3) bucket. This is an AWS Public Dataset located at `arn:aws:s3:::allen-brain-observatory` in region `us-west-2`. Launching a Jupyter notebook instance on AWS will allow you to access the complete dataset without having to download anything locally. This includes around 80 TB of raw data files, which are not accessible via the AllenSDK. The only drawback is that you'll need to pay for the time that your instance is running—but this can still be economical in many cases. A brief overview of this approach can be found <a href='#Accessing-data-on-AWS'>below</a>.
#
# A third option is to directly download the data via **api.brain-map.org**. This should be used only as a last resort if the other options are broken or are not available to you. Instructions for this can be found <a href='#Direct-download-via-api.brain-map.org'>at the end of this tutorial</a>.

# + [markdown] papermill={"duration": 0.006367, "end_time": "2023-11-30T06:23:22.934556", "exception": false, "start_time": "2023-11-30T06:23:22.928189", "status": "completed"} pycharm={"name": "#%% md\n"}
# ## Using the AllenSDK to retrieve data

# + [markdown] papermill={"duration": 0.006475, "end_time": "2023-11-30T06:23:22.947343", "exception": false, "start_time": "2023-11-30T06:23:22.940868", "status": "completed"} pycharm={"name": "#%% md\n"}
# Most users will want to access data via the AllenSDK. This requires nothing more than a Python interpreter and some free disk space to store the data locally.
#
# How much data is there? If you want to download the complete dataset (58 experiments), you'll need 855 GB of space, split across the following files:
#
# 1. CSV files containing information about sessions, probes, channels and units (58.1 MB)
# 2. NWB files containing spike times, behavior data, and stimulus information for each session (146.5 GB total, min file size = 1.7 GB, max file size = 3.3 GB)
# 3. NWB files containing LFP data for each probe (707 GB total, min file size = 0.9 GB, max file size = 2.7 GB)
#
# Before downloading the data, you must decide where the `manifest.json` file lives. This file serves as the map that guides the `EcephysProjectCache` object to the file locations.
#
# When you initialize a local cache for the first time, it will create the manifest file at the path that you specify. This file lives in the same directory as the rest of the data, so make sure you put it somewhere that has enough space available. 
#
# When you need to access the data in subsequent analysis sessions, you should point the `EcephysProjectCache` object to an _existing_ `manifest.json` file; otherwise, it will try to re-download the data in a new location.
#
# To get started with this approach, first take care of the necessary imports:

# + papermill={"duration": 4.216733, "end_time": "2023-11-30T06:23:27.170380", "exception": false, "start_time": "2023-11-30T06:23:22.953647", "status": "completed"} pycharm={"name": "#%%\n"}
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# + [markdown] papermill={"duration": 0.006475, "end_time": "2023-11-30T06:23:27.183824", "exception": false, "start_time": "2023-11-30T06:23:27.177349", "status": "completed"} pycharm={"name": "#%% md\n"}
# Next, we'll specify the location of the manifest file. If you're creating a cache for the first time, this file won't exist yet, but it _must_ be placed in an existing data directory. Remember to choose a location that has plenty of free space available.

# + papermill={"duration": 0.012164, "end_time": "2023-11-30T06:23:27.202538", "exception": false, "start_time": "2023-11-30T06:23:27.190374", "status": "completed"} pycharm={"name": "#%%\n"} tags=["parameters"]
output_dir = '/media/user/Data/Yann/VisualCoding_Neuropix' # must be updated to a valid directory in your filesystem
DOWNLOAD_COMPLETE_DATASET = True

# + papermill={"duration": 0.011422, "end_time": "2023-11-30T06:23:27.238949", "exception": false, "start_time": "2023-11-30T06:23:27.227527", "status": "completed"} pycharm={"name": "#%%\n"}
manifest_path = os.path.join(output_dir, "manifest.json")

# + [markdown] papermill={"duration": 0.006453, "end_time": "2023-11-30T06:23:27.251971", "exception": false, "start_time": "2023-11-30T06:23:27.245518", "status": "completed"} pycharm={"name": "#%% md\n"}
# Now we can create the cache object, specifying both the local storage directory (the `manifest_path`) and the remote storage location (the Allen Institute data warehouse).

# + papermill={"duration": 0.012288, "end_time": "2023-11-30T06:23:27.270770", "exception": false, "start_time": "2023-11-30T06:23:27.258482", "status": "completed"} pycharm={"name": "#%%\n"}
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# + [markdown] papermill={"duration": 0.006512, "end_time": "2023-11-30T06:23:27.283867", "exception": false, "start_time": "2023-11-30T06:23:27.277355", "status": "completed"} pycharm={"name": "#%% md\n"}
# This will prepare the cache to download four files:
#
# 1. `sessions.csv` (7.8 kB)
# 2. `probes.csv` (27.0 kB)
# 3. `channels.csv` (6.6 MB)
# 4. `units.csv` (51.4 MB)
#
# Each one contains a table of information related to its file name. If you're using the AllenSDK, you won't have to worry about how these files are formatted. Instead, you'll load the relevant data using specific accessor functions: `get_session_table()`, `get_probes()`, `get_channels()`, and `get_units()`. These functions return a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html?highlight=dataframe) containing a row for each item and a column for each metric.
#
# If you are analyzing data without using the AllenSDK, you can load the data using your CSV file reader of choice. However, please be aware the columns in the original file do not necessarily match what's returned by the AllenSDK, which may combine information from multiple files to produce the final DataFrame.
#
# Let's take a closer look at what's in the `sessions.csv` file:

# + papermill={"duration": 229.59866, "end_time": "2023-11-30T06:27:16.889009", "exception": false, "start_time": "2023-11-30T06:23:27.290349", "status": "completed"} pycharm={"name": "#%%\n"}
sessions = cache.get_session_table()

print('Total number of sessions: ' + str(len(sessions)))

sessions.head()

# + [markdown] papermill={"duration": 0.006816, "end_time": "2023-11-30T06:27:16.902711", "exception": false, "start_time": "2023-11-30T06:27:16.895895", "status": "completed"} pycharm={"name": "#%% md\n"}
# The `sessions` DataFrame provides a high-level overview of the Neuropixels Visual Coding dataset. The index column is a unique ID, which serves as a key for accessing the physiology data for each session. The other columns contain information about:
#
# - the session type (i.e., which stimulus set was shown?)
# - the age, sex, and genotype of the mouse (in this dataset, there's only one session per mouse)
# - the number of probes, channels, and units for each session
# - the brain structures recorded (CCFv3 acronyms)
#
# If we want to find all of recordings from male Sst-Cre mice that viewed the Brain Observatory 1.1 stimulus and contain units from area LM, we can use the following query:

# + papermill={"duration": 0.021114, "end_time": "2023-11-30T06:27:16.930525", "exception": false, "start_time": "2023-11-30T06:27:16.909411", "status": "completed"} pycharm={"name": "#%%\n"}
filtered_sessions = sessions[(sessions.sex == 'M') & \
                             (sessions.full_genotype.str.find('Sst') > -1) & \
                             (sessions.session_type == 'brain_observatory_1.1') & \
                             (['VISl' in acronyms for acronyms in 
                               sessions.ecephys_structure_acronyms])]

filtered_sessions.head()

# + [markdown] papermill={"duration": 0.006943, "end_time": "2023-11-30T06:27:16.944460", "exception": false, "start_time": "2023-11-30T06:27:16.937517", "status": "completed"} pycharm={"name": "#%% md\n"}
# The `filtered_sessions` table contains the three sessions that meet these criteria.
#
# The code above uses standard syntax for filtering pandas DataFrames. If this is unfamiliar to you, we strongly recommend reading through the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/). The AllenSDK makes heavy use of pandas objects, so we don't have to come up with our own functions for working with tabular data.
#
# Let's take a look at another DataFrame, extracted from the `probes.csv` file.

# + papermill={"duration": 0.652684, "end_time": "2023-11-30T06:27:17.604010", "exception": false, "start_time": "2023-11-30T06:27:16.951326", "status": "completed"} pycharm={"name": "#%%\n"}
probes = cache.get_probes()

print('Total number of probes: ' + str(len(probes)))

probes.head()

# + [markdown] papermill={"duration": 0.007107, "end_time": "2023-11-30T06:27:17.618986", "exception": false, "start_time": "2023-11-30T06:27:17.611879", "status": "completed"} pycharm={"name": "#%% md\n"}
# The `probes` DataFrame contains information about the Neuropixels probes used across all recordings. Each row represents one probe from one recording session, even though the physical probes may have been used in multiple sessions. Some of the important columns are:
#
# - `ecephys_session_id`: the index column of the `sessions` table
# - `sampling_rate`: the sampling rate (in Hz) for this probe's spike band; note that each probe has a unique sampling rate around 30 kHz. The small variations in sampling rate across probes can add up to large offsets over time, so it's critical to take these differences into account. However, all of the data you will interact with has been pre-aligned to a common clock, so this value is included only for reference purposes.
# - `lfp_sampling_rate`: the sampling rate (in Hz) for this probe's LFP band NWB files, after 2x downsampling from the original rate of 2.5 kHz
# - `name`: the probe name is assigned based on the location of the probe on the recording rig. This is useful to keep in mind because probes with the same name are always targeted to the same cortical region and enter the brain from the same angle (`probeA` = AM, `probeB` = PM, `probeC` = V1, `probeD` = LM, `probeE` = AL, `probeF` = RL). However, the targeting is not always accurate, so the actual recorded region may be different.
# - `phase`: the data may have been generated by one of two "phases" of Neuropixels probes. **3a** = prototype version; **PXI** = publicly available version ("Neuropixels 1.0"). The two phases should be equivalent from the perspective of data analysis, but there may be differences in the noise characteristics between the two acquisition systems.
# - `channel_count`: the number of channels with spikes or LFP data (maximum = 384)
#
# The `channels.csv` file contains information about each of these channels.

# + papermill={"duration": 0.573066, "end_time": "2023-11-30T06:27:18.199099", "exception": false, "start_time": "2023-11-30T06:27:17.626033", "status": "completed"} pycharm={"name": "#%%\n"}
channels = cache.get_channels()

print('Total number of channels: ' + str(len(channels)))

channels.head()

# + [markdown] papermill={"duration": 0.00743, "end_time": "2023-11-30T06:27:18.214540", "exception": false, "start_time": "2023-11-30T06:27:18.207110", "status": "completed"} pycharm={"name": "#%% md\n"}
# The most important columns in the `channels` DataFrame concern each channel's location in physical space. Each channel is associated with a location along the probe shank (`probe_horizontal_position` and `probe_vertical_position`), and may be linked to a coordinate in the Allen Common Coordinate framework (if CCF registration is available for that probe).
#
# The information about channel location will be merged into the `units` DataFrame, which is loaded from `units.csv`:

# + papermill={"duration": 0.486721, "end_time": "2023-11-30T06:27:18.708587", "exception": false, "start_time": "2023-11-30T06:27:18.221866", "status": "completed"} pycharm={"name": "#%%\n"}
units = cache.get_units()

print('Total number of units: ' + str(len(units)))

# + [markdown] papermill={"duration": 0.007425, "end_time": "2023-11-30T06:27:18.724012", "exception": false, "start_time": "2023-11-30T06:27:18.716587", "status": "completed"} pycharm={"name": "#%% md\n"}
# This DataFrame contains metadata about the available units across all sessions. By default, the AllenSDK applies some filters to this table and only returns units above a particular quality threshold.
#
# The default filter values are as follows:
#
# - `isi_violations` < 0.5
# - `amplitude_cutoff` < 0.1
# - `presence_ratio` > 0.9
#
# For more information about these quality metrics and how to interpret them, please refer to [this tutorial](./ecephys_quality_metrics.ipynb).
#
# If you want to see _all_ of the available units, it's straightfoward to disable the quality metrics filters when retrieving this table: 

# + papermill={"duration": 0.542013, "end_time": "2023-11-30T06:27:19.273351", "exception": false, "start_time": "2023-11-30T06:27:18.731338", "status": "completed"} pycharm={"name": "#%%\n"}
units = cache.get_units(amplitude_cutoff_maximum = np.inf,
                        presence_ratio_minimum = -np.inf,
                        isi_violations_maximum = np.inf)

print('Total number of units: ' + str(len(units)))

# + [markdown] papermill={"duration": 0.007662, "end_time": "2023-11-30T06:27:19.289297", "exception": false, "start_time": "2023-11-30T06:27:19.281635", "status": "completed"} pycharm={"name": "#%% md\n"}
# As you can see, the number of units has increased substantially, but some fraction of these units will be incomplete or highly contaminated. Understanding the meaning of these metrics is a critical part of analyzing the Neuropixels dataset, so we strongly recommend learning how to interpret them correctly.
#
# In addition to the quality metrics, there are a number of stimulus-specific metrics that are computed for each unit. These are not downloaded by default, but are accessed via a separate SDK function:

# + papermill={"duration": 182.184477, "end_time": "2023-11-30T06:30:21.481560", "exception": false, "start_time": "2023-11-30T06:27:19.297083", "status": "completed"} pycharm={"name": "#%%\n"}
analysis_metrics1 = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')

analysis_metrics2 = cache.get_unit_analysis_metrics_by_session_type('functional_connectivity')

print(str(len(analysis_metrics1)) + ' units in table 1')
print(str(len(analysis_metrics2)) + ' units in table 2')

# + [markdown] papermill={"duration": 0.007509, "end_time": "2023-11-30T06:30:21.496762", "exception": false, "start_time": "2023-11-30T06:30:21.489253", "status": "completed"} pycharm={"name": "#%% md\n"}
# This will download two additional files, `brain_observatory_1.1_analysis_metrics.csv` and `functional_connectivity_analysis_metrics.csv`, and load them as pandas DataFrames. Note that the total length of these DataFrames is around 40k units, because the default quality metric thresholds have been applied.
#
# To load _all_ of the available units, and create one giant table of metrics, you can use the following code:

# + papermill={"duration": 3.025239, "end_time": "2023-11-30T06:30:24.529448", "exception": false, "start_time": "2023-11-30T06:30:21.504209", "status": "completed"} pycharm={"name": "#%%\n"}
analysis_metrics1 = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1', amplitude_cutoff_maximum = np.inf,
                                                          presence_ratio_minimum = -np.inf,
                                                          isi_violations_maximum = np.inf)

analysis_metrics2 = cache.get_unit_analysis_metrics_by_session_type('functional_connectivity', amplitude_cutoff_maximum = np.inf,
                                                          presence_ratio_minimum = -np.inf,
                                                          isi_violations_maximum = np.inf)

all_metrics = pd.concat([analysis_metrics1, analysis_metrics2], sort=False)

print(str(len(all_metrics)) + ' units overall')

# + [markdown] papermill={"duration": 0.007501, "end_time": "2023-11-30T06:30:24.545054", "exception": false, "start_time": "2023-11-30T06:30:24.537553", "status": "completed"} pycharm={"name": "#%% md\n"}
# The length of this DataFrame should match that of the `units` DataFrame we retrieved earlier. A few things to note about this DataFrame:
#
# - The unit analysis metrics DataFrame _also_ includes all quality metrics, so it's a superset of the `units` DataFrame
# - Since some of the stimuli in the `brain_observatory_1.1` session are not present in the `functional_connectivity` session, many of the data points in the unit analysis metrics DataFrame will be filled with `nan` values

# + [markdown] papermill={"duration": 0.037424, "end_time": "2023-11-30T06:30:24.590113", "exception": false, "start_time": "2023-11-30T06:30:24.552689", "status": "completed"} pycharm={"name": "#%% md\n"}
# ### Accessing data for individual sessions
#
# Assuming you've found a session you're interested in analyzing in more detail, it's now time to download the data. This is as simple as calling `cache.get_session_data()`, with the `session_id` as input. This method will check the cache for an existing NWB file and, if it's not present, will automatically download it for you.
#
# Each NWB file can be upwards of 2 GB, so please be patient while it's downloading!
#
# As an example, let's look at one of the sessions we selected earlier, disabling the default unit quality metrics filters:

# + papermill={"duration": 215.084249, "end_time": "2023-11-30T06:33:59.681957", "exception": false, "start_time": "2023-11-30T06:30:24.597708", "status": "completed"} pycharm={"name": "#%%\n"}
session = cache.get_session_data(filtered_sessions.index.values[0],
                                 isi_violations_maximum = np.inf,
                                 amplitude_cutoff_maximum = np.inf,
                                 presence_ratio_minimum = -np.inf
                                )

print([attr_or_method for attr_or_method in dir(session) if attr_or_method[0] != '_'])

# + [markdown] papermill={"duration": 0.077525, "end_time": "2023-11-30T06:33:59.839817", "exception": false, "start_time": "2023-11-30T06:33:59.762292", "status": "completed"} pycharm={"name": "#%% md\n"}
# As you can see, the `session` object has a lot of attributes and methods that can be used to access the underlying data in the NWB file. Most of these will be touched on in other tutorials, but for now we will look at the only one that is capable of triggering additional data downloads, `get_lfp()`.
#
# In general, each NWB file is meant to be a self-contained repository of data for one recording session. However, for the Neuropixels data, we've broken with convention a bit in order to store LFP data in separate files. If we hadn't done this, analyzing one session would require an initial 15 GB file download. Now, the session is broken up in to ~2 GB chunks..
#
# Once you have created a `session` object, downloading the LFP data is simple (but may be slow):

# + papermill={"duration": 179.544711, "end_time": "2023-11-30T06:36:59.462220", "exception": false, "start_time": "2023-11-30T06:33:59.917509", "status": "completed"} pycharm={"name": "#%%\n"}
probe_id = session.probes.index.values[0]

lfp = session.get_lfp(probe_id)

# + [markdown] papermill={"duration": 0.130023, "end_time": "2023-11-30T06:36:59.724222", "exception": false, "start_time": "2023-11-30T06:36:59.594199", "status": "completed"} pycharm={"name": "#%% md\n"}
# Tips for analyzing LFP data can be found in [this tutorial](./ecephys_lfp_analysis.ipynb).

# + [markdown] papermill={"duration": 0.130385, "end_time": "2023-11-30T06:36:59.986077", "exception": false, "start_time": "2023-11-30T06:36:59.855692", "status": "completed"} pycharm={"name": "#%% md\n"}
# ### Downloading the complete dataset
#
# Analyzing one session at a time is nice, but in many case you'll want to be able to query across the whole dataset. To fill your cache with all available data, you can use a `for` loop like the one below. Note that we've added some checks to ensure that the complete file is present, in case the download has been interrupted due to an unreliable connection.
#
# Before running this code, please make sure that you have enough space available in your cache directory. You'll need around 855 GB for the whole dataset, and 147 GB if you're not downloading the LFP data files.

# + papermill={"duration": 0.140528, "end_time": "2023-11-30T06:37:00.259967", "exception": false, "start_time": "2023-11-30T06:37:00.119439", "status": "completed"} pycharm={"name": "#%%\n"}
if DOWNLOAD_COMPLETE_DATASET:
    for session_id, row in sessions.iterrows():

        truncated_file = True
        directory = os.path.join(output_dir + '/session_' + str(session_id))

        while truncated_file:
            session = cache.get_session_data(session_id)
            try:
                print(session.specimen_name)
                truncated_file = False
            except OSError:
                shutil.rmtree(directory)
                print(" Truncated spikes file, re-downloading")

        for probe_id, probe in session.probes.iterrows():

            print(' ' + probe.description)
            truncated_lfp = True

            while truncated_lfp:
                try:
                    lfp = session.get_lfp(probe_id)
                    truncated_lfp = False
                except OSError:
                    fname = directory + '/probe_' + str(probe_id) + '_lfp.nwb'
                    os.remove(fname)
                    print("  Truncated LFP file, re-downloading")
                except ValueError:
                    print("  LFP file not found.")
                    truncated_lfp = False

# + [markdown] papermill={"duration": 0.130696, "end_time": "2023-11-30T06:37:00.524750", "exception": false, "start_time": "2023-11-30T06:37:00.394054", "status": "completed"} pycharm={"name": "#%% md\n"}
# ## Accessing data on AWS

# + [markdown] papermill={"duration": 0.130405, "end_time": "2023-11-30T06:37:00.786078", "exception": false, "start_time": "2023-11-30T06:37:00.655673", "status": "completed"} pycharm={"name": "#%% md\n"}
# If you want to analyze the data without downloading anything to your local machine, you can use the AllenSDK on AWS.
#
# Follow [these instructions](https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS) to launch a Jupyter notebook. Then, simply point to the existing manifest file in the Allen Institute's S3 bucket, and all of the data will be immediately available:

# + papermill={"duration": 0.167183, "end_time": "2023-11-30T06:37:01.086515", "exception": false, "start_time": "2023-11-30T06:37:00.919332", "status": "completed"} pycharm={"name": "#%%\n"}
cache = EcephysProjectCache(manifest=manifest_path)

# + [markdown] papermill={"duration": 0.132476, "end_time": "2023-11-30T06:37:01.352339", "exception": false, "start_time": "2023-11-30T06:37:01.219863", "status": "completed"} pycharm={"name": "#%% md\n"}
# Once your cache is initialized, you can create the `sessions` table, load individual `session` objects, and access LFP data using the same commands described above.
#
# Additional tutorials specific to using AWS are coming soon.

# + [markdown] papermill={"duration": 0.130782, "end_time": "2023-11-30T06:37:01.615168", "exception": false, "start_time": "2023-11-30T06:37:01.484386", "status": "completed"} pycharm={"name": "#%% md\n"}
# ## Direct download via api.brain-map.org

# + [markdown] papermill={"duration": 0.130764, "end_time": "2023-11-30T06:37:01.877139", "exception": false, "start_time": "2023-11-30T06:37:01.746375", "status": "completed"} pycharm={"name": "#%% md\n"}
# Some people have reported issues downloading the files via the AllenSDK (the connection is extremely slow, or gets interrupted frequently). If this applies to you, you can try downloading the files via HTTP requests sent to **api.brain-map.org**. This approach is not recommended, because you will have to manually keep track of the file locations. But if you're doing analysis that doesn't depend on the AllenSDK (e.g., in Matlab), this may not matter to you.
#
# You can follow the steps below to retrieve the URLs for all of the NWB files in this dataset.

# + papermill={"duration": 0.166857, "end_time": "2023-11-30T06:37:02.174570", "exception": false, "start_time": "2023-11-30T06:37:02.007713", "status": "completed"} pycharm={"name": "#%%\n"}
from allensdk.brain_observatory.ecephys.ecephys_project_api.utilities import build_and_execute
from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import RmaEngine
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

rma_engine = RmaEngine(scheme="http", host="api.brain-map.org")

# + papermill={"duration": 0.793805, "end_time": "2023-11-30T06:37:03.100741", "exception": false, "start_time": "2023-11-30T06:37:02.306936", "status": "completed"} pycharm={"name": "#%%\n"}
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()


# + papermill={"duration": 10.111752, "end_time": "2023-11-30T06:37:13.345613", "exception": false, "start_time": "2023-11-30T06:37:03.233861", "status": "completed"} pycharm={"name": "#%%\n"}
def retrieve_link(session_id):
    
    well_known_files = build_and_execute(
        (
        "criteria=model::WellKnownFile"
        ",rma::criteria,well_known_file_type[name$eq'EcephysNwb']"
        "[attachable_type$eq'EcephysSession']"
        r"[attachable_id$eq{{session_id}}]"
        ),
        engine=rma_engine.get_rma_tabular, 
        session_id=session_id
    )
    
    return 'http://api.brain-map.org/' + well_known_files['download_link'].iloc[0]

download_links = [retrieve_link(session_id) for session_id in sessions.index.values]

_ = [print(link) for link in download_links]

# + [markdown] papermill={"duration": 0.142373, "end_time": "2023-11-30T06:37:13.621529", "exception": false, "start_time": "2023-11-30T06:37:13.479156", "status": "completed"} pycharm={"name": "#%% md\n"}
# `download_links` is a list of 58 links that can be used to download the NWB files for all available sessions. Clicking on the links above should start the download automatically.
#
# Please keep in mind that you'll have to move these files to the appropriate sub-directory once the download is complete. The `EcephysProjectCache` object expects the following directory structure:
#
# ```
# cache_dir/
# # +-- manifest.json               
# # +-- session_<id>/    
# ¦   +-- session_<id>.nwb
# # +-- session_<id>/
# ¦   +-- session_<id>.nwb
# # +-- session_<id>/
# ¦   +-- session_<id>.nwb
#
# ```
#
# If you aren't interested in using the `EcephysProjectCache` object to keep track of what you've downloaded, you can create a `session` object just by passing a path to an NWB file:

# + papermill={"duration": 0.136313, "end_time": "2023-11-30T06:37:13.917966", "exception": false, "start_time": "2023-11-30T06:37:13.781653", "status": "completed"} pycharm={"name": "#%%\n"}
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession

# nwb_path = '/mnt/nvme0/ecephys_cache_dir_10_31/session_721123822/session_721123822.nwb'

# session = EcephysSession.from_nwb_path(nwb_path, api_kwargs={
#         "amplitude_cutoff_maximum": np.inf,
#         "presence_ratio_minimum": -np.inf,
#         "isi_violations_maximum": np.inf
#     })

# + [markdown] papermill={"duration": 0.131136, "end_time": "2023-11-30T06:37:14.180603", "exception": false, "start_time": "2023-11-30T06:37:14.049467", "status": "completed"} pycharm={"name": "#%% md\n"}
# This will load the data for one session, without applying the default unit quality metric filters. Everything will be available except the LFP data, because the `get_lfp()` method can only find the associated LFP files if you're using the `EcephysProjectCache` object.
#
# To obtain similar links for the LFP files, you can use the following code:

# + papermill={"duration": 52.526123, "end_time": "2023-11-30T06:38:06.837588", "exception": false, "start_time": "2023-11-30T06:37:14.311465", "status": "completed"} pycharm={"name": "#%%\n"}
def retrieve_lfp_link(probe_id):

    well_known_files = build_and_execute(
        (
            "criteria=model::WellKnownFile"
            ",rma::criteria,well_known_file_type[name$eq'EcephysLfpNwb']"
            "[attachable_type$eq'EcephysProbe']"
            r"[attachable_id$eq{{probe_id}}]"
        ),
        engine=rma_engine.get_rma_tabular, 
        probe_id=probe_id
    )

    if well_known_files.shape[0] != 1:
        return 'file for probe ' + str(probe_id) + ' not found'
        
    return 'http://api.brain-map.org/' + well_known_files.loc[0, "download_link"]

probes = cache.get_probes()

download_links = [retrieve_lfp_link(probe_id) for probe_id in probes.index.values]

_ = [print(link) for link in download_links]
