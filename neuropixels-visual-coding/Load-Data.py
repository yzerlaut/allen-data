# %%
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache\
      import EcephysProjectCache
manifest_path=os.path.join(\
                os.path.expanduser('~'), 'Downloads',
                'ecephys_cache_dir', 'manifest.json')
cache = EcephysProjectCache.from_warehouse(\
                                manifest=manifest_path)
all_sessions = cache.get_session_table()

print('Total number of sessions: ' + str(len(all_sessions)))

# all_sessions.head()

# %%
all_sessions.head()

# %%
# find sessions with photo-tagging of INs
sessions = all_sessions[\
            all_sessions.full_genotype.str.find('IRES-Cre') > -1]
print(30*'--'+'\n--> Number of sessions with IN activity : ' + str(len(sessions))+'\n'+30*'--')

# %%

class Data:

      def __init__(self, 
                   session_id, sample_id,
                   tstart=None, tstop=None):

            self.session_id = session_id
            self.sample_id = sample_id 

            if tstart is None:
                 self.load()
            else:
                  self.tstart, self.tstop = tstart, tstop

      def save(self):
            data = {}
            for key in ['tstart', 'tstop',
                        'running_speed', 'pupil_area']:
                 if hasattr(self, key):
                      data[key] = getattr(self, key)
            for loc in ['VISp']:
                 if hasattr(self, key):
                      data['raster_%s' % loc] = getattr(self, 'raster_%s' % loc)
                      data['lfp_%s' % loc] = getattr(self, 'lfp_%s' % loc)
            np.save(os.path.join('data', 
                                 'SpontAct_session-%i_sample-%i.npy'\
                                          % (self.session_id, self.sample_id)),
                                          data)
      def load(self):
            data = np.load(os.path.join('data', 
                                 'SpontAct_session-%i_sample-%i.npy'\
                                          % (self.session_id, self.sample_id)),
                                          allow_pickle=True).item()
            for key in data:
                 setattr(self, key, data[key])



for session_id in sessions.index[:1]:

    sample_index = 0 # we restart the sample index

    session = cache.get_session_data(session_id)
    stim_table = session.get_stimulus_table()
    spont_periods = stim_table[stim_table['stimulus_name']=='spontaneous']

    for tstart, tstop in zip(spont_periods.start_time,
                             spont_periods.stop_time):

        if (tstop-tstart)>120:
            # more than 2min spont activity

            sample_index += 1

            data = Data(session_id, sample_index, tstart, tstop)

            # let's fetch the running speed
            cond = (session.running_speed.end_time.values>tstart) &\
                (session.running_speed.start_time.values<tstop)

            # t_running_speed = .5*(session.running_speed.start_time.values[cond]+\
            #                            session.running_speed.end_time.values[cond])
            data.running_speed = session.running_speed.velocity[cond]

            pupil = session.get_pupil_data()
            if pupil is not None:
                data.pupil_area = np.pi*pupil['pupil_height'].values/2.*pupil['pupil_width'].values/2.

            # let's fetch the isolated single units in V1
            units = session.units[\
                session.units.ecephys_structure_acronym == 'VISp'] # V1==VISp
            data.raster_VISp = []
            for i in units.index:
                cond = (session.spike_times[i]>=tstart) & (session.spike_times[i]<tstop)
                data.raster_VISp.append(session.spike_times[i][cond])

            # let's fetch the V1 probe --> always on "probeC"
            probe_id = session.probes[session.probes.description == 'probeC'].index.values[0]

            # -- let's fetch the lfp data for that probe and that session --
            # let's fetch the all the channels falling into V1 domain
            # data.V1_channel_ids = session.channels[(session.channels.probe_id == probe_id) & \
            #               (session.channels.ecephys_structure_acronym.isin(['VISp']))].index.values

            # limit LFP to desired times and channels
            # N.B. "get_lfp" returns a subset of all channels above
            data.lfp_VISp = session.get_lfp(probe_id).sel(time=slice(tstart, tstop),
                                                              channel=14)
            # self.Nchannels_V1 = len(self.lfp_slice_V1.channel) # store number of channels with LFP in V1
            # self.lfp_sampling_rate = session.probes.lfp_sampling_rate[probe_id] # keeping track of sampling rate
            
            print('data successfully loaded in %.1fs' % (time.time()-tic))
            data.save()

# all_sessions.full_genotype.str.find('Pvalb-IRES-Cre')

# %%

import sys
sys.path.append('..')
import plot_tools as pt
pt.set_style('dark')

data = Data(715093703, 3)

fig, AX = pt.figure(axes=(3,1), ax_scale=(2,1))

# raster
for i, times in enumerate(data.raster_VISp):
     AX[0].plot(times, i+0*times, 'o')

# lfp
t = np.linspace(data.tstart, data.tstop, len(data.lfp_VISp))
AX[-1].plot(t-t[0], data.lfp_VISp)
pt.draw_bar_scales(AX[-1], 
                   Ybar=10, Ybar_label='10',
                   Xbar=10, Xbar_label='10s')

# running speed 
t = np.linspace(data.tstart, data.tstop, len(data.running_speed))
AX[-1].plot(t-t[0], data.running_speed)
pt.draw_bar_scales(AX[-1], 
                   Ybar=10, Ybar_label='10cm/s',
                   Xbar=10, Xbar_label='10s')

for ax in AX:
     ax.axis('off')
     ax.set_xlim([data.tstart, data.tstop])

# %%
