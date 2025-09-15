# %%
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache\
      import EcephysProjectCache
manifest_path=os.path.join(\
                os.path.expanduser('~'), 'Downloads',
                'ecephys_cache_dir', 'manifest.json')
cache = EcephysProjectCache.from_warehouse(\
                                manifest=manifest_path)
sessions = cache.get_session_table()

print('Total number of sessions: ' + str(len(sessions)))

sessions.head()
# %%
