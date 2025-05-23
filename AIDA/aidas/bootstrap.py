import logging;
import os;
import importlib;

import aidacommon.aidaConfig;
from aidacommon.aidaConfig import AConfig;
from aidacommon import rop;
import aidas.dmro as dmro;
import aidas.aidas as aidas;
import aidas.scheduler as scheduler;

def bootstrap():

##    try:
##        configfile = os.environ['AIDACONFIG'];
##    except KeyError:
##        raise EnvironmentError('Environment variable AIDACONFIG is not set.');
##
##        # Check if the config file exists.
##    if (not os.path.isfile(configfile)):
##        raise FileNotFoundError("Error configuration file {} not found.".format(configfile));
##
##        # Load the configuration settings.
##    config = configparser.ConfigParser();
##    config.read(configfile);
##    defaultConfig = config['DEFAULT'];
##    serverConfig = config['AIDASERVER'];
##    AConfig.DATABASEPORT = serverConfig.getint('DATABASEPORT', defaultConfig['DATABASEPORT']);
##    AConfig.DATABASEADAPTER = serverConfig.get('DATABASEADAPTER', defaultConfig['DATABASEADAPTER']);
##    AConfig.LOGLEVEL = serverConfig.get('LOGLEVEL', defaultConfig['LOGLEVEL']);
##    AConfig.LOGFILE = servermy_python_udfConfig.get('LOGFILE', defaultConfig['LOGFILE']);
##    AConfig.RMIPORT = serverConfig.getint('RMIPORT', defaultConfig['RMIPORT']);
##    AConfig.CONNECTIONMANAGERPORT = serverConfig.getint('CONNECTIONMANAGERPORT', defaultConfig['CONNECTIONMANAGERPORT']);
##    udfType = serverConfig.get('UDFTYPE', defaultConfig['UDFTYPE']);
##    AConfig.UDFTYPE = UDFTYPE.TABLEUDF if (udfType == 'TABLEUDF') else UDFTYPE.VIRTUALTABLE;
##
##    # Setup the logging mechanism.
##    if (AConfig.LOGLEVEL == 'DEBUG'):
##        logl = logging.DEBUG;
##    elif (AConfig.LOGLEVEL == 'WARNING'):
##        logl = logging.WARNING;
##    elif (AConfig.LOGLEVEL == 'ERROR'):
##        logl = logging.ERROR;
##    else:
##        logl = logging.INFO;
##    logging.basicConfig(filename=AConfig.LOGFILE, level=logl);
##    logging.info('AIDA: Bootstrap procedure aidas_bootstrap starting...');

    aidacommon.aidaConfig.loadConfig('AIDASERVER');

    logging.info("Connection manager port : " + str(aidacommon.aidaConfig.AConfig.CONNECTIONMANAGERPORT))
    logging.info("RMI port : " + str(aidacommon.aidaConfig.AConfig.RMIPORT))

    # Initialize the DMRO repository.
    try:
        dmro.DMROrepository('aidasys');
    except Exception as e:
        logging.exception(e);

    import aidasys;

    # Startup the remote object manager for RMI.
    robjMgr = rop.ROMgr.getROMgr('', AConfig.RMIPORT, True);
    aidasys.robjMgr = robjMgr;
    
    schMgr = scheduler.ScheduleManager.getScheduleManager();
    aidasys.schMgr = schMgr;

    # Start the connection manager.    # Get the module and class name separated out for the database adapter that we need to load.
    dbAdapterModule, dbAdapterClass = os.path.splitext(AConfig.DATABASEADAPTER);
    dbAdapterClass = dbAdapterClass[1:];
    dmod = importlib.import_module(dbAdapterModule);
    dadapt = getattr(dmod, dbAdapterClass);
    logging.info('AIDA: Loading database adapter {} for connection manager'.format(dadapt))
    conMgr = aidas.ConnectionManager.getConnectionManager(dadapt);
    aidasys.conMgr = conMgr;

def callback(args):
    logging.info('AIDA: Callback called.')
