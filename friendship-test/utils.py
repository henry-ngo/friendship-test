# utils.py

import numpy as np
#from astropy import units as u # May be necesary because of object?
from astropy.time import Time
#from astropy.coordinates import Angle, ICRS  # May be necssary because of object?
import de421
import jplephem
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Object that stores info for recreating plots
class track_data(object):
    # Plot metadata
    objname=''
    dt=0.
    nmctrials=0.
    startdate=0.
    enddate=0.
    refID=0.
    nobs=0
    npriorobs=0
    npharoobs=0
    # Tracks and their 1+2 sigma limits
    datearr=0.
    sep=0.
    sep_1sig=0. # 2D array, second index indicates min/max
    sep_2sig=0.
    pa=0.
    pa_1sig=0.
    pa_2sig=0.
    # measured dates/positions (and where they would be if they were bg)
    obsdates=0.
    meas_sep=0.
    meas_sep_err=0.
    bg_sep=0.
    meas_pa=0.
    meas_pa_err=0.
    bg_pa=0.
    # pharo measured dates/positions (and where they would be if they were bg)
    pharo_obsdates=0.
    pharo_sep=0.
    pharo_sep_err=0.
    pharo_bg_sep=0.
    pharo_pa=0.
    pharo_pa_err=0.
    pharo_bg_pa=0.
    # prior dates/positions (and where they would be if they were bg)
    prior_obsdates=0.
    prior_sep=0.
    prior_sep_err=0.
    prior_bg_sep=0.
    prior_pa=0.
    prior_pa_err=0.
    prior_bg_pa=0.
    prior_cite=''

    # The class "constructor"/"initializer"
    def __init__(self,objname,dt,nmctrials,startdate,enddate,refID,datearr,sep,sep_1sig,sep_2sig,pa,pa_1sig,pa_2sig,obsdates,meas_sep,meas_sep_err,bg_sep,meas_pa,meas_pa_err,bg_pa,pharo_obsdates,pharo_sep,pharo_sep_err,pharo_bg_sep,pharo_pa,pharo_pa_err,pharo_bg_pa,prior_obsdates,prior_sep,prior_sep_err,prior_bg_sep,prior_pa,prior_pa_err,prior_bg_pa,prior_cite):
        self.objname=objname
        self.dt=dt
        self.nmctrials=nmctrials
        self.startdate=startdate
        self.enddate=enddate
        self.refID=refID
        self.nobs=obsdates.size
        self.npharoobs=pharo_obsdates.size
        self.npriorobs=prior_obsdates.size
        self.datearr=datearr
        self.sep=sep
        self.sep_1sig=sep_1sig
        self.sep_2sig=sep_2sig
        self.pa=pa
        self.pa_1sig=pa_1sig
        self.pa_2sig=pa_2sig
        self.obsdates=obsdates
        self.meas_sep=meas_sep
        self.meas_sep_err=meas_sep_err
        self.bg_sep=bg_sep
        self.meas_pa=meas_pa
        self.meas_pa_err=meas_pa_err
        self.bg_pa=bg_pa
        self.pharo_obsdates=pharo_obsdates
        self.pharo_sep=pharo_sep
        self.pharo_sep_err=pharo_sep_err
        self.pharo_bg_sep=pharo_bg_sep
        self.pharo_pa=pharo_pa
        self.pharo_pa_err=pharo_pa_err
        self.pharo_bg_pa=pharo_bg_pa
        self.prior_obsdates=prior_obsdates
        self.prior_sep=prior_sep
        self.prior_sep_err=prior_sep_err
        self.prior_bg_sep=prior_bg_sep
        self.prior_pa=prior_pa
        self.prior_pa_err=prior_pa_err
        self.prior_bg_pa=prior_bg_pa
        self.prior_cite=prior_cite

# gets the background position of star (equivalent to background_position.pro)
# But in millarcseconds!
def get_bg_pos(sep,pa,pm_ra,pm_dec,ra,dec,parallax,ref_epoch,final_epoch,PA_near_zero=False,parallax_offset=True):
    """
    Takes separation and pos ang of stars, the proper motion, reference epoch and final epoch
    and returns final separation [mas] and position angle [deg]
    INPUTS:
    sep             -- separation between stars, in milliarcseconds
    pa              -- postion angle in degrees E of N
    pm_ra           -- proper motion of star, in RA, in milliarcseconds
    pm_dec          -- proper motion of star, in DEC, in milliarcseconds
    ra              -- RA of star in degrees
    dec             -- DEC of star in degrees
    parallax        -- parallax of star (in milliarcseconds; 1000.0*1/dist_pc)
    ref_epoch       -- Epoch at which ALL sep, pa, ra, dec were measured [decimal date]
    final_epoch     -- Epoch at which to compute output sep/pa [decimal date]
    KEYWORD INPUTS:
    PA_near_zero    -- Output PAs in interval (-180,180) instead of (0,360)
    parallax_offset -- Consider effect of Earth's motion (parallax) [default TRUE]
    OUTPUTS:
    out_sep         -- separation at final_epoch, in milliarcseconds
    out_pa          -- position angle at final_epoch, in degrees
    """
    # Get initial x,y separations
    x0=sep*np.sin(np.radians(pa))
    y0=sep*np.cos(np.radians(pa))

    # Get adjustment from proper motion (pm values in milliarcseconds/yr)
    # (-1 since companion moves opposite direction to star)
    dx_pm = (final_epoch-ref_epoch)*pm_ra*-1.0
    dy_pm = (final_epoch-ref_epoch)*pm_dec*-1.0
    new_x = x0 + dx_pm
    new_y = y0 + dy_pm

    # Get adjustment from parallax (if keyword set)
    if parallax_offset:
        # Get offsets at ref and final epochs
        # (unlike parallax.pro, we use real parallax here so no need to adjust for it later)
        (ref_raoff,ref_decoff)=get_parallax_offset(ra,dec,parallax,ref_epoch)
        (out_raoff,out_decoff)=get_parallax_offset(ra,dec,parallax,final_epoch)
        # Apply offset: first subtract off reference epoch to get true position, then add parallax offset
        # (However, apply -1 factor due to background object moving in opposite direction)
        new_x = (new_x+ref_raoff)  - out_raoff
        new_y = (new_y+ref_decoff) - out_decoff

    # Compute new sep and pa
    out_sep=np.sqrt(new_x**2.0+new_y**2.0)
    out_pa=np.degrees(np.arctan2(new_x,new_y)) # in degrees now
    out_pa=np.mod(out_pa,360.0) # between 0 and 360deg
    if PA_near_zero:
        if out_pa >180:
            out_pa-=360.0 # now between -180 to +180
    return (out_sep,out_pa)


# gets the offset due to parallax of Earth (equivalent to parallax.pro)
def get_parallax_offset(ra,dec,parallax,epoch,ra_units='detector'):
    """
    Takes the star's ra, dec, parallax, and a given epoch.
    Returns the parallax shift in star's position, in arcseconds (sexagesimal seconds for raoff; decimal seconds for decoff)
    INPUTS:
    ra          -- RA position of star, in degrees
    dec         -- DEC position of star, in degrees
    parallax    -- parallax of star, in MILLIarcseconds (1000.0*1/distance_pc)
    epoch       -- epoch (decimal years) to compute parallax offset (scalar or monotonically increasing vector)
    ra_units    -- USE EITHER:
                   'time': raoff is in seconds of time in RA (need to multiply by 15 to get same units as decoff)
                   'detector': [default] 1 unit displacement same in both horizontal/vertical direction
                               (factor of cos(dec) applied to raoff; multiply by 15 to get same units)
    OUTPUTS:
    raoff       -- parallax shift in star's position, in milliarcseconds
    decoff      -- parallax shift in star's position, in milliarcseconds
    """
    # Convert from deg to rad
    ra_rad=np.radians(ra)
    dec_rad=np.radians(dec)

    # Use jplephem with de421 to determine position of Earth geocenter wrt SS Barycenter
    eph=jplephem.Ephemeris(de421)
    t=Time(epoch,format='jyear',scale='ut1')
    JD=t.jd
    barycenter = eph.position('earthmoon', JD)
    moonvector = eph.position('moon', JD)
    earthPos = (barycenter - moonvector * eph.earth_share) / 1.49597871e+8
    usex=np.float(earthPos[0])
    usey=np.float(earthPos[1])
    usez=np.float(earthPos[2])

    # Compute offsets (in milliarcseconds)
    delta_alpha = parallax * 1./np.cos(dec_rad) * (usex*np.sin(ra_rad) - usey*np.cos(ra_rad))
    delta_delta = parallax * (usex*np.cos(ra_rad)*np.sin(dec_rad) + usey*np.sin(ra_rad)* np.sin(dec_rad) - usez*np.cos(dec_rad))
    # Convert RA seconds as necessary
    if ra_units=='time':
        delta_alpha = delta_alpha / 15.0
    elif ra_units=='detector':
        delta_alpha = delta_alpha * np.cos(dec_rad)
    else:
        raise Exception('ra_units set to unhandled option: {}'.format(ra_units))

    # Set outputs
    raoff=delta_alpha
    decoff=delta_delta

    return (raoff,decoff)

# Returns the value corresponding to given confidence interval
# Assumes 1D array!
def get_ci(array,ci):
    array_sort=np.sort(array)
    ind_low=np.floor((0.5-ci/2.0)*array.size).astype(np.int)
    ind_high=np.ceil((0.5+ci/2.0)*array.size).astype(np.int)
    return (array_sort[ind_low],array_sort[ind_high])

# Function to create and save the plot (o=track_data object)
def make_plot(o,outfilename,titlestring,titleloc,no_prior=False,no_pharo=False,no_bg=False,for_poster=False):
    # "for_poster" makes plots slightly differently formatted and coloured
    if for_poster:
        myplotfonts={'size': 36}
        max_n_xtics=4
        max_n_ytics=4
        xlabel_fontsize=24
        ylabel_fontsize=24
        our_data_fmt='ko'
    else: # this is for paper
        myplotfonts={'size': 24}
        max_n_xtics=4
        max_n_ytics=6
        xlabel_fontsize=24
        ylabel_fontsize=24
        our_data_fmt='ko'
    matplotlib.rc('font',**myplotfonts)
    # Use thicker axes border
    matplotlib.rc('axes', linewidth=2.0)
    # Get a few important parameters
    nobs=o.obsdates.size # not used
    # Create figure
    fig = plt.figure()
    # sep subplot
    ax_sep = fig.add_subplot(2,1,1)
    # Make title label (on SEP SUBPLOT) -- only if not making plots for poster!
    if not for_poster:
        if titleloc=='left':
            ax_sep.text(0.03,0.95,titlestring,weight='bold',horizontalalignment='left',verticalalignment='top',transform=ax_sep.transAxes)
        elif titleloc=='center':
            ax_sep.text(0.50,0.95,titlestring,weight='bold',horizontalalignment='center',verticalalignment='top',transform=ax_sep.transAxes)
        elif titleloc=='right':
            ax_sep.text(0.97,0.95,titlestring,weight='bold',horizontalalignment='right',verticalalignment='top',transform=ax_sep.transAxes)
        elif titleloc=='none':
            pass # don't make a title!
        else:
            raise Exception('Invalid value for titleloc: {}'.format(titleloc))
    # Plot BG track and 68+95 confidence intervals (use zorder to place shaded stuff behind the rest)
    if not no_bg:
        ax_sep.plot(o.datearr,o.sep,'k-',linewidth=2)
        ax_sep.fill_between(o.datearr,o.sep_1sig[:,0],o.sep_1sig[:,1],color='0.5',zorder=-1)
        ax_sep.fill_between(o.datearr,o.sep_2sig[:,0],o.sep_2sig[:,1],color='0.8',zorder=-2)
    # Plot constant sep value (with errorbar)
    #ax_sep.axhline(y=np.mean(o.meas_sep),color='black',linestyle=':')
    ax_sep.axhline(y=o.meas_sep[o.refID],color='black',linestyle=':')
    ax_sep.axhline(y=o.meas_sep[o.refID]+o.meas_sep_err[o.refID],color='black',linestyle=':')
    ax_sep.axhline(y=o.meas_sep[o.refID]-o.meas_sep_err[o.refID],color='black',linestyle=':')
    # For each measurement, plot actual position (filled) and if they were bg (open)
    if not for_poster and not no_bg:
        ax_sep.plot(o.obsdates,o.bg_sep,'ko',ms=10,mew=2,mfc='None')
    ax_sep.errorbar(o.obsdates,o.meas_sep,yerr=o.meas_sep_err,fmt=our_data_fmt,ms=10)
    ax_sep.set_xlim([o.startdate,o.enddate])
    #ax_sep.set_ylim([2440,2530]) manual adjustment for a specific plot
    # Plot PHARO measurements, if they exist AND ``no_pharo'' is not set:
    if o.npharoobs > 0 and not no_pharo:
        if not no_bg:
            ax_sep.plot(o.pharo_obsdates,o.pharo_bg_sep,'k^',ms=10,mew=2,mfc='None')
        ax_sep.errorbar(o.pharo_obsdates,o.pharo_sep,yerr=o.pharo_sep_err,fmt='k^',ms=10)
    # Plot prior measurements, if they exist AND ``no_prior'' is not set:
    if o.npriorobs > 0 and not no_prior:
        if not no_bg:
            ax_sep.plot(o.prior_obsdates,o.prior_bg_sep,'ks',ms=10,mew=2,mfc='None')
        ax_sep.errorbar(o.prior_obsdates,o.prior_sep,yerr=o.prior_sep_err,fmt='ks',ms=10)
    # Format x axis tics (turn off labels, but keep tic marks)
    ax_sep.xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_n_xtics,integer=True)) # sets xtics at nice locations
    ax_sep.xaxis.set_major_formatter(ticker.NullFormatter()) # do not show xtic labels
    ax_sep.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=4))
    # Format y axis tics (prune edge for ganged plots)
    ax_sep.yaxis.set_major_locator(ticker.MaxNLocator(nbins=max_n_ytics,prune='both')) # set ytics at nice locations
    ax_sep.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax_sep.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    if not for_poster:
        ax_sep.set_ylabel('Sep [mas]',fontsize=ylabel_fontsize)
    # Increase length and thickness of tics
    ax_sep.tick_params('both',length=10,width=2.0,which='major')
    ax_sep.tick_params('both',length=5,width=2.0,which='minor')
    # PA subplot
    ax_pa=fig.add_subplot(2,1,2)
    # Plot BG track and 68+95 confidence intervals (use zorder to place shaded stuff behind the rest)
    if not no_bg:
        ax_pa.plot(o.datearr,o.pa,'k-',linewidth=2)
        ax_pa.fill_between(o.datearr,o.pa_1sig[:,0],o.pa_1sig[:,1],color='0.5',zorder=-1)
        ax_pa.fill_between(o.datearr,o.pa_2sig[:,0],o.pa_2sig[:,1],color='0.7',zorder=-2)
    # Plot constant PA value (with errorbar)
    #ax_pa.axhline(y=np.mean(o.meas_pa),color='black',linestyle=':')
    ax_pa.axhline(y=o.meas_pa[o.refID],color='black',linestyle=':')
    ax_pa.axhline(y=o.meas_pa[o.refID]+o.meas_pa_err[o.refID],color='black',linestyle=':')
    ax_pa.axhline(y=o.meas_pa[o.refID]-o.meas_pa_err[o.refID],color='black',linestyle=':')
    # For each measurement, plot actual position (filled) and if they were bg (open)
    if not for_poster and not no_bg:
        ax_pa.plot(o.obsdates,o.bg_pa,'ko',ms=10,mew=2,mfc='None')
    ax_pa.errorbar(o.obsdates,o.meas_pa,yerr=o.meas_pa_err,fmt=our_data_fmt,ms=10)
    ax_pa.set_xlim([o.startdate,o.enddate])
    # Plot PHARO measurements, if they exist AND ``no_pharo'' is not set:
    if o.npharoobs > 0 and not no_pharo:
        if not no_bg:
            ax_pa.plot(o.pharo_obsdates,o.pharo_bg_pa,'k^',ms=10,mew=2,mfc='None')
        ax_pa.errorbar(o.pharo_obsdates,o.pharo_pa,yerr=o.pharo_pa_err,fmt='k^',ms=10)
    # Plot prior measurements, if they exist AND ``no_prior'' is not set:
    if o.npriorobs > 0 and not no_prior:
        if not no_bg:
            ax_pa.plot(o.prior_obsdates,o.prior_bg_pa,'ks',ms=10,mew=2,mfc='None')
        ax_pa.errorbar(o.prior_obsdates,o.prior_pa,yerr=o.prior_pa_err,fmt='ks',ms=10)
    # Format x axis tics
    ax_pa.xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_n_xtics,integer=True)) # sets xtics at nice locations
    ax_pa.xaxis.set_major_formatter(ticker.FormatStrFormatter('%4d')) # format for xtic labels
    ax_pa.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=4))
    if not for_poster:
        ax_pa.set_xlabel('Date [yr]',fontsize=xlabel_fontsize)
    # Format y axis tics (prune edges for ganged plots)
    ax_pa.yaxis.set_major_locator(ticker.MaxNLocator(nbins=max_n_ytics,prune='both')) # set ytics at nice locations
    ax_pa.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax_pa.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    if not for_poster:
        ax_pa.set_ylabel('PA [deg]',fontsize=ylabel_fontsize)
    # Increase length and thickness of tics
    ax_pa.tick_params('both',length=10,width=2.0,which='major')
    ax_pa.tick_params('both',length=5,width=2.0,which='minor')
    # Adjust so that everything fits and plots are stacked
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.001) # no horizontal space between figures
    fig.savefig(outfilename) # png for display
    fig.savefig(outfilename.replace('.png','.svg'),format='svg',dpi=1200) # for plotting

# Output the data used in making plot
def make_data_table(o,outfilename):
    f=open(outfilename,'w')
    # Write: Object name, obs-date, separation, error, PA, error, citation
    for ii in range(0,o.npriorobs):
        f.write('{} & {:7.2f} & ${:6.1f} \pm {:6.1f}$ & ${:5.1f} \pm {:5.1f}$ & {}\\\\\n'.format(o.objname,o.prior_obsdates[ii],o.prior_sep[ii],o.prior_sep_err[ii],o.prior_pa[ii],o.prior_pa_err[ii],o.prior_cite[ii]))
    for jj in range(0,o.nobs):
        if o.meas_pa_err[jj] < 0.1:
            f.write('{} & {:7.2f} & ${:6.1f} \pm {:6.1f}$ & ${:5.2f} \pm {:5.2f}$ & {}\\\\\n'.format(o.objname,o.obsdates[jj],o.meas_sep[jj],o.meas_sep_err[jj],o.meas_pa[jj],o.meas_pa_err[jj],'this work'))
        else:
            f.write('{} & {:7.2f} & ${:6.1f} \pm {:6.1f}$ & ${:5.1f} \pm {:5.1f}$ & {}\\\\\n'.format(o.objname,o.obsdates[jj],o.meas_sep[jj],o.meas_sep_err[jj],o.meas_pa[jj],o.meas_pa_err[jj],'this work'))
    f.close()
