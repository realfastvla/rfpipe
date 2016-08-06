from . import state, source, search
import distributed
import rtpipe

# need to design state initialization as an avalanch of decisions set by initial state
# 1) initial, minimal state defines either parameters for later use or fixes final state
# 2) read metadata from observation for given scan (if final value not yet set)
# 3) run functions to set state (sets hashable state)
# 4) may also set convenience attibutes
# 5) run data processing for given segment

# these should be modified to change state based on input
# - nsegments or dmarr + memory_limit + metadata => segmenttimes
# - dmarr or dm_parameters + metadata => dmarr
# - uvoversample + npix_max + metadata => npixx, npixy

# maybe this all goes in state.py outside of class?


def run(datasource, paramfile, version=2):
    """ Run whole pipeline """

    st = state.State(paramfile=paramfile, verison=version)

    if datatype(datasource) == 'sdm':
        apply_metadata(st, sdmfile)

    # learn distributed for this part
    if 'image' in st.searchtype:
        search.imaging(st)


def apply_metadata(st, sdmfile):
    """ Run all functions to apply metadata to transform initial state """

    source.parsesdm(sdmfile, st)
    state.set_dmgrid(st)
    state.set_imagegrid(st)
    state.set_segments(st)


def pipeline(sdmfile, scan):
    ex = distributed.Executor('nmpost029:8786')

    st = ex.submit(rtpipe.RT.set_pipeline, sdmfile, scan, memory_limit=3)

    nsegment = d['nsegment']
    data = ex.map(rtpipe.parsesdm.read_bdf_segment(st, range(nsegment)))
    uvw = ex.map(rtpipe.parsesdm.get_uvw_segment(st, range(nsegment)))
