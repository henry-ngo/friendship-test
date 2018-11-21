# Creates figures with measured astrometry and background tracks
# Henry Ngo, June 2014
# main driver routine + usage() and read_input_file()

import os,sys,getopt
import numpy as np
from utils import *
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import ICRS, Angle
import pickle

def usage():
    print ("Usage: run_friendship_test.py [-h] [--help] [--recompute] [--poster] [--PAzero] [--dt value] [--nmctrials value] [--titleloc pos] [--titlestr value] [--nopharo] [--noprior] [--nobg] objname startdate enddate")
    print ("Note: if a pickle file containing computed tracks is available, code will use this instead")
    print ("Required arguments:")
    print ("    objname: name of object; expect inputfile: input/objname.in")
    print ("    startdate: first date to plot/compute (years)")
    print ("    enddate: last date to plot/compute (years)")
    print ("Optional arguments:")
    print ("    -h, --help: prints this help")
    print ("    --recompute: recomputes existing bgtrack object instead of loading from file")
    print ("    --poster: formats plots differently, for display on a poster!")
    print ("    --PAzero: Use for PAs close to 0, will force angles to be -180 to +180 instead")
    print ("    --dt value: resolution of dates plotted, in years [default: 0.05]")
    print ("    --nmctrials value: number of monte carlo draws for confidence intervals [default: 200]")
    print ("    --titleloc pos: position of title string one of: (l[eft],c[entre],r[ight],n[one]) [default: left]")
    print ("    --titlestr value: use a custom title")
    print ("    --nopharo: do not plot our PHARO measurements [default: False]")
    print ("    --noprior: do not plot measurements from prior studies [default: False]")
    print ("    --nobg: do not plot background track (e.g. no proper motion data) [default: False]")


def read_input_file(INPUTFILE):
    """
    ra, dec: in HH:MM:SS and DD:MM:SS, respectively, as strings
    """
    try:
        f=open(INPUTFILE,'r')
    except:
        raise Exception("file {0} cannot be opened for reading.".format(INPUTFILE))
    # Read the file
    nHeadLines=5
    wholeFile = f.read() # Entire file is read in as a string
    f.close() # closes the file
    fileLines = np.array(wholeFile.splitlines()) # Each element = 1 line of file
    # Remove lines that are blank or are commented out (starting with '#')
    lengths=np.array([len(x) for x in fileLines]) # get line lengths
    keep_ind=np.where(lengths>0)
    fileLines=fileLines[keep_ind] # gets rid of blank lines
    char1=np.array([x[0] for x in fileLines]) # get first character
    keep_ind=np.where(char1!='#')
    fileLines=fileLines[keep_ind] # gets rid of commented lines
    # Compute number of dates given
    nLines = fileLines.size
    nDates = nLines-nHeadLines
    # Parse RA/DEC
    posarr=np.array(fileLines[0].split())
    if posarr.size==2:
        if ':' in posarr[0]: # assume HH:MM:SS DD:MM:SS notation
            targpos=ICRS(ra=Angle(posarr[0]+' hours'),dec=Angle(posarr[1]+' degrees'))
        else: # assume decimal DEGREES
            targpos=ICRS(ra=Angle(posarr[0]+' degrees'),dec=Angle(posarr[1]+' degrees'))
    elif posarr.size==6: # HH MM SS DD MM SS
        targpos=ICRS(ra=Angle('{}:{}:{} hours'.format(*posarr[0:3])),dec=Angle('{}:{}:{} degrees'.format(*posarr[3:6])))
    else:
        raise Exception('Input line 1 should contain either 2 or 6 entries')
    # Output RA/DEC as HH:MM:SS and DD:MM:SS strings
    ra=targpos.ra.deg
    dec=targpos.dec.deg
    # Parse RA/DEC error ellipse and compute positional errors
    pos_err_arr=np.array(fileLines[1].split()).astype(np.float)
    if pos_err_arr.size < 2 or pos_err_arr.size > 3:
        raise Exception('Input line 2 must contain 2 or 3 entries!')
    # Compute 1D uncertainties from error ellipse
    (ra_err,dec_err)=parse_error_ellipse(pos_err_arr)
    # Convert ra_err,dec_err from mas to degrees (3600arcsec=1deg)
    ra_err=(ra_err/1000.0)/3600.0
    dec_err=(dec_err/1000.0)/3600.0
    # Parse distance/parallax
    distarr=np.array(fileLines[2].split()).astype(np.float)
    if distarr.size!=3:
        raise Exception('Input line 3 must contain three entries')
    parallax=distarr[0]
    parallax_err_high=distarr[1]
    parallax_err_low=distarr[2]
    parallax_err=parse_asymmetric_error(parallax_err_high,parallax_err_low)
    # Parse proper motion (mu_ra*cos(dec), mu_dec)
    pmarr=np.array(fileLines[3].split()).astype(np.float)
    if pmarr.size != 2:
        raise Exception('Input line 4 must contain two entries')
    pmra=pmarr[0]
    pmdec=pmarr[1]
    # Parse proper motion error ellipse and compute pm errors
    err_arr=np.array(fileLines[4].split()).astype(np.float)
    if err_arr.size < 2 or err_arr.size > 3:
        raise Exception('Input line 5 must contain 2 or 3 entries!')
    # Compute 1D uncertainties from error ellipse
    (pmra_err,pmdec_err)=parse_error_ellipse(err_arr)
    # Loop through remaining lines to read astrometry measurements
    obsdate=np.zeros(nDates)
    sep=np.zeros(nDates)
    sep_err=np.zeros(nDates)
    pa=np.zeros(nDates)
    pa_err=np.zeros(nDates)
    for ii in range(0,nDates):
        astr_arr=np.array(fileLines[nHeadLines+ii].split()).astype(np.float)
        # If date is in JD, convert to Julian years (e.g. 2012.34)
        if astr_arr[0] > 2100:
            t=Time(astr_arr[0],format='jd',scale='ut1')
            obsdate[ii]=t.jyear
        else:
            obsdate[ii]=astr_arr[0]
        # Store the rest of the measurements
        sep[ii],sep_err[ii],pa[ii],pa_err[ii]=astr_arr[1:]
    # Determine which measurement to be the reference value (minimum separation error)
    refMeas=np.argmin(sep_err)

    return (ra,ra_err,dec,dec_err,parallax,parallax_err,pmra,pmra_err,pmdec,pmdec_err,obsdate,sep,sep_err,pa,pa_err,refMeas)

## Read input for PHARO/prior information!
def read_other_input_file(INPUTFILE):
    """ INPUTFILE should have format:
    [YYYY-MM-DD] [SEP_MAS] [SEP_ERR_MAS] [PA_DEG] [PA_ERR_DEG] [JONES_ET_AL._(2001)]
    """
    try:
        f=open(INPUTFILE,'r')
    except:
        raise Exception("file {0} cannot be opened for reading.".format(INPUTFILE))
    # Read the file
    wholeFile = f.read() # Entire file is read in as a string
    f.close() # closes the file
    fileLines = np.array(wholeFile.splitlines()) # Each element = 1 line of file
    # Remove lines that are blank or are commented out (starting with '#')
    lengths=np.array([len(x) for x in fileLines]) # get line lengths
    keep_ind=np.where(lengths>0)
    fileLines=fileLines[keep_ind] # gets rid of blank lines
    char1=np.array([x[0] for x in fileLines]) # get first character
    keep_ind=np.where(char1!='#')
    fileLines=fileLines[keep_ind] # gets rid of commented lines
    # Compute number of observations given
    nLines = fileLines.size
    # Read prior measurements
    obsdate=np.zeros(nLines)
    sep=np.zeros(nLines)
    sep_err=np.zeros(nLines)
    pa=np.zeros(nLines)
    pa_err=np.zeros(nLines)
    citation=np.zeros(nLines).astype(np.str)
    for ii in range(0,nLines):
        # Convert date from YYYY-MM-DD to julian years
        input_date=fileLines[ii].split()[0]
        t=Time(input_date,format='iso',scale='ut1')
        obsdate[ii]=t.jyear
        # Get the astrometry measurements
        sep[ii],sep_err[ii],pa[ii],pa_err[ii]=np.array(fileLines[ii].split()[1:5]).astype(np.float)
        # Get the citation information
        citation[ii]=fileLines[ii].split()[5].replace('_',' ')
    return (obsdate,sep,sep_err,pa,pa_err,citation)

# Quick routine to convert a SIMBAD error ellipse to standard deviations in each of 2 parameters
# Returns sigma=(sigma_ra,sigma_dec) given error ellipse (a,b,posang)
# See http://cdsweb.u-strasbg.fr/simbad/guide/errell.htx
def parse_error_ellipse(params):
    a=params[0]
    b=params[1]
    try:
        pa=np.radians(params[2])
    except:
        pa=np.radians(90.0)
    sigma_ra=np.sqrt(np.sin(pa)**2.0*a**2.0+np.cos(pa)**2.0*b**2.0)
    sigma_dec=np.sqrt(np.cos(pa)**2.0*a**2.0+np.sin(pa)**2.0*b**2.0)
    sigmas=np.array([sigma_ra,sigma_dec])
    return sigmas

# Quick routine to convert asymmetrical error bars into a single error bar
# Currently, takes average!
def parse_asymmetric_error(error_high,error_low):
    return 0.5*(error_high+error_low)

# Driver routine
def main():
    # Parse Input
    try:
        (opts,args)=getopt.getopt(sys.argv[1:],"h",["help","nstars=","recompute","poster","PAzero","dt=","nmctrials=","titleloc=","titlestr=","nopharo","noprior","nobg"])
    except getopt.GetoptError as err:
        print (str(err)) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    # Set default values
    nstars=2
    RECOMPUTE=False # recomputes bgtracks even if pickle file exists
    POSTER_FORMAT=False
    PAzero=False # set when PA is close to 0 (so PAs goes from -180 to +180)
    epoch_dt=0.05 # 0.05 works well  (time between plotted/computed epochs)
    ntrials=200 # 100 works well (number of MC trials to produce per epoch)
    titlelocstring='left'
    custom_title_string=''
    NOPHARO=False # if set, plots won't show PHARO data
    NOPRIOR=False # if set, plots won't show prior data
    NOBG=False # if set, plots won't show bg track
    # Parse options
    for o,a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--nstars="):
            try:
                nstars=np.int(a)
            except:
                raise Exception('Invalid value for nstars: {}'.format(a))
        elif o in ("--recompute"):
            RECOMPUTE=True
        elif o in ("--poster"):
            POSTER_FORMAT=True
        elif o in ("--PAzero"):
            PAzero=True
        elif o in ("--dt="):
            try:
                epoch_dt=np.float(a)
            except:
                raise Exception('Invalid value for dt: {}'.format(a))
        elif o in ("--nmctrials="):
            try:
                ntrials=np.int(a)
            except:
                raise Exception('Invalid value for nmctrials: {}'.format(a))
        elif o in ("--titleloc="):
            if a.lower() in ("l","left"):
                titlelocstring='left'
            elif a.lower() in ("c","center"):
                titlelocstring='center'
            elif a.lower() in ("r","right"):
                titlelocstring='right'
            elif a.lower() in ("n","none"):
                titlelocstring='none'
            else:
                raise Exception('Invalid value for titleloc: {}'.format(a))
        elif o in ("--titlestr="):
            custom_title_string=a
        elif o in ("--nopharo"):
            NOPHARO=True
        elif o in ("--noprior"):
            NOPRIOR=True
        elif o in ("--nobg"):
            NOBG=True

    # Parse arguments
    if len(args) < 3:
        usage()
        sys.exit(2)
    try:
        startdate=np.float(args[1])
    except:
        raise Exception('Invalid value for startdate: {}'.format(args[2]))
    try:
        enddate=np.float(args[2])
    except:
        raise Exception('Invalid value for enddate: {}'.format(args[2]))
    nepochs=np.ceil((enddate-startdate)/epoch_dt).astype(np.int)

    # Set up important files and directories
    OBJNAME=args[0]
    INPUTFILE='input/'+OBJNAME+'.in'
    PHARO_INPUTFILE='input/'+OBJNAME+'_PHARO.in'
    PRIOR_INPUTFILE='input/'+OBJNAME+'_prior_studies.in'
    if not os.path.exists(INPUTFILE):
        raise Exception('INPUTFILE not found: {}'.format(INPUTFILE))
    if POSTER_FORMAT:
        PLOTDIR='plots_poster/'
    else:
        PLOTDIR='plots/'
    plotfile=PLOTDIR+OBJNAME+'.png'
    PICKLEDIR='pickled_tracks/'
    picklefile=PICKLEDIR+OBJNAME+'.pickle.dat'
    TABLEDIR='astrometry_tables/'
    tablefile=TABLEDIR+OBJNAME+'.tex'
    necessary_dirs=(PLOTDIR,PICKLEDIR,TABLEDIR)
    for mydir in necessary_dirs:
        if not os.path.exists(mydir):
            os.mkdir(mydir)

    # Read input file
    (ra,ra_err,dec,dec_err,parallax,parallax_err,pmra,pmra_err,pmdec,pmdec_err,obsdates,sep,sep_err,pa,pa_err,refMeas)=read_input_file(INPUTFILE)
    nobs=sep.size # number of measurements
    # If it exists, read PHARO input data
    if os.path.exists(PHARO_INPUTFILE):
        (pharo_obsdates,pharo_sep,pharo_sep_err,pharo_pa,pharo_pa_err,pharo_cite)=read_other_input_file(PHARO_INPUTFILE)
        # Force PAs to be from -180 to +180 if PAzero is set
        if PAzero:
            pharo_pa[np.where(pharo_pa>180.0)]-=360.0
    else: # Otherwise, set all to empty zero length arrays
        pharo_obsdates=np.array([])
        pharo_sep=np.array([])
        pharo_sep_err=np.array([])
        pharo_pa=np.array([])
        pharo_pa_err=np.array([])
        pharo_cite=np.array([])
    npharoobs=pharo_sep.size
    # If it exists, read the prior input data
    if os.path.exists(PRIOR_INPUTFILE):
        (prior_obsdates,prior_sep,prior_sep_err,prior_pa,prior_pa_err,prior_cite)=read_other_input_file(PRIOR_INPUTFILE)
        # Force PAs to be from -180 to +180 if PAzero is set
        if PAzero:
            prior_pa[np.where(prior_pa>180.0)]-=360.0
    else: # Otherwise, set all to empty zero length arrays
        prior_obsdates=np.array([])
        prior_sep=np.array([])
        prior_sep_err=np.array([])
        prior_pa=np.array([])
        prior_pa_err=np.array([])
        prior_cite=np.array([])
    npriorobs=prior_sep.size
    # If not set to recompute and picklefile exists, check to make sure it's correct
    if not RECOMPUTE and os.path.exists(picklefile):
        # Check if pickle file is correct:
        fpickle=open(picklefile,'rb')
        bgtest=pickle.load(fpickle)
        fpickle.close()
        if epoch_dt != bgtest.dt or ntrials != bgtest.nmctrials or startdate != bgtest.startdate or enddate != bgtest.enddate or refMeas != bgtest.refID or nobs != bgtest.nobs or npriorobs != bgtest.npriorobs:
            RECOMPUTE=True # No match, force recomputation
        del bgtest # delete to avoid confusion
    # If set to compute (or forced because picklefile wrong) or no picklefile exists, then recompute
    if RECOMPUTE or not os.path.exists(picklefile):
        # Monte Carlo the background trajectory
        mc_ra=np.random.normal(loc=ra,scale=ra_err,size=ntrials)
        mc_dec=np.random.normal(loc=dec,scale=dec_err,size=ntrials)
        mc_pmra=np.random.normal(loc=pmra,scale=pmra_err,size=ntrials)
        mc_pmdec=np.random.normal(loc=pmdec,scale=pmdec_err,size=ntrials)
        mc_parallax=np.random.normal(loc=parallax,scale=parallax_err,size=ntrials)
        mc_sep=np.zeros((nepochs,ntrials))
        mc_pa=np.zeros((nepochs,ntrials))
        for ii in range(0,nepochs):
            mc_sep[ii,:]=np.random.normal(loc=sep[refMeas],scale=sep_err[refMeas],size=ntrials)
            mc_pa[ii,:]=np.random.normal(loc=pa[refMeas],scale=pa_err[refMeas],size=ntrials)

        # Create arrays to store the 68% and 95% confidence limits for sep/pa at each epoch
        mc68_bg_track_sep=np.zeros((nepochs,2)) # min and max values
        mc95_bg_track_sep=np.zeros((nepochs,2)) # min and max values
        mc68_bg_track_pa=np.zeros((nepochs,2)) # min and max values
        mc95_bg_track_pa=np.zeros((nepochs,2)) # min and max values

        # Compute background tracks from reference measurement (using best fit value AND monte carlo values)
        datearr=np.linspace(startdate,enddate,nepochs)
        bg_track_sep=np.zeros(nepochs)
        bg_track_pa=np.zeros(nepochs)
        sys.stdout.write('Computing tracks for {}: '.format(OBJNAME))
        for ii in range(0,nepochs):
            # Here's the track using the data
            aa,bb=get_bg_pos(sep[refMeas],pa[refMeas],pmra,pmdec,ra,dec,parallax,obsdates[refMeas],datearr[ii],PA_near_zero=PAzero,parallax_offset=True)
            bg_track_sep[ii]=aa
            bg_track_pa[ii]=bb
            # Here's the Monte Carlo tracks
            all_mc_sep=np.zeros(ntrials)
            all_mc_pa=np.zeros(ntrials)
            for jj in range(0,ntrials):
                aa,bb=get_bg_pos(mc_sep[refMeas,jj],mc_pa[refMeas,jj],mc_pmra[jj],mc_pmdec[jj],mc_ra[jj],mc_dec[jj],mc_parallax[jj],obsdates[refMeas],datearr[ii],PA_near_zero=PAzero,parallax_offset=True)
                all_mc_sep[jj]=aa
                all_mc_pa[jj]=bb
            # And get the confidence intervals
            min68,max68=get_ci(all_mc_sep,0.68)
            mc68_bg_track_sep[ii,:]=np.array([min68,max68])
            min95,max95=get_ci(all_mc_sep,0.95)
            mc95_bg_track_sep[ii,:]=np.array([min95,max95])
            min68,max68=get_ci(all_mc_pa,0.68)
            mc68_bg_track_pa[ii,:]=np.array([min68,max68])
            min95,max95=get_ci(all_mc_pa,0.95)
            mc95_bg_track_pa[ii,:]=np.array([min95,max95])
            # Print progress counter
            percentDone=(100.0*(ii+1)/nepochs)
            if ii < 1:
                sys.stdout.write('{:3.0f}% '.format(percentDone))
            else:
                sys.stdout.write('\b\b\b\b\b{:3.0f}% '.format(percentDone))
            sys.stdout.flush()

        # Get sep/pa of the other observations, IF THEY WERE BG OBJECTS (i.e. Brendan's open symbols)
        sep_if_bg=np.zeros(nobs)
        pa_if_bg=np.zeros(nobs)
        pharo_sep_if_bg=np.zeros(npharoobs)
        pharo_pa_if_bg=np.zeros(npharoobs)
        prior_sep_if_bg=np.zeros(npriorobs)
        prior_pa_if_bg=np.zeros(npriorobs)
        for kk in range(0,nobs):
            (sep_if_bg[kk],pa_if_bg[kk])=get_bg_pos(sep[refMeas],pa[refMeas],pmra,pmdec,ra,dec,parallax,obsdates[refMeas],obsdates[kk],PA_near_zero=PAzero,parallax_offset=True)
        for kk in range(0,npharoobs):
            (pharo_sep_if_bg[kk],pharo_pa_if_bg[kk])=get_bg_pos(sep[refMeas],pa[refMeas],pmra,pmdec,ra,dec,parallax,obsdates[refMeas],pharo_obsdates[kk],PA_near_zero=PAzero,parallax_offset=True)
        for kk in range(0,npriorobs):
            (prior_sep_if_bg[kk],prior_pa_if_bg[kk])=get_bg_pos(sep[refMeas],pa[refMeas],pmra,pmdec,ra,dec,parallax,obsdates[refMeas],prior_obsdates[kk],PA_near_zero=PAzero,parallax_offset=True)

        # Create an object for these calculations and pickle them!
        bgtrack=track_data(OBJNAME,epoch_dt,ntrials,startdate,enddate,refMeas,datearr,bg_track_sep,mc68_bg_track_sep,mc95_bg_track_sep,bg_track_pa,mc68_bg_track_pa,mc95_bg_track_pa,obsdates,sep,sep_err,sep_if_bg,pa,pa_err,pa_if_bg,pharo_obsdates,pharo_sep,pharo_sep_err,pharo_sep_if_bg,pharo_pa,pharo_pa_err,pharo_pa_if_bg,prior_obsdates,prior_sep,prior_sep_err,prior_sep_if_bg,prior_pa,prior_pa_err,prior_pa_if_bg,prior_cite)
        fpickle=open(picklefile,'wb')
        pickle.dump(bgtrack,fpickle)
        fpickle.close()
    else:
        print ('Load up pre-computed track for {} from {}'.format(OBJNAME,picklefile))
        fpickle=open(picklefile,'rb')
        bgtrack=pickle.load(fpickle)
        fpickle.close()
    # Set plot title
    if custom_title_string=='':
        titlestring=bgtrack.objname
    else:
        titlestring=custom_title_string
    # Make the plot!
    make_plot(bgtrack,plotfile,titlestring,titlelocstring,no_pharo=NOPHARO,no_prior=NOPRIOR,no_bg=NOBG,for_poster=POSTER_FORMAT)
    print ('| Wrote plot file: {}'.format(plotfile))
    print ('| Wrote plot file: {}'.format(plotfile.replace('.png','.svg')))
    make_data_table(bgtrack,tablefile)

if __name__=="__main__":
    main()

# subplot configurations
#pos1a = [.12,.5,.48,.85]
#pos1b = [.12,.15,.48,.5]
#pos2 = [.52,.24,.9,.76]
