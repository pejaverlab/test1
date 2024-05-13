import configparser



class ConfigModule:
    
    def __init__(self):
        self.alpha = None
        self.B = None
        self.discountonesided = None
        self.windowclinvarpoints = None
        self.windowgnomadfraction = None
        self.emulate_tavtigian = None
        self.emulate_pejaver = None
        self.gaussian_smoothing = None

    def load_config(self, filepath):

        cfg = configparser.ConfigParser()
        cfg.read(filepath)
        cfg.sections()
        
        #c = int(cfg['tuningparameters']['c'])
        self.B = int(cfg['tuningparameters']['B'])
        self.discountonesided = float(cfg['tuningparameters']['discountonesided'])
        self.windowclinvarpoints = int(cfg['tuningparameters']['windowclinvarpoints'])
        self.windowgnomadfraction = float(cfg['tuningparameters']['windowgnomadfraction'])
        self.gaussian_smoothing = cfg['smoothing'].getboolean('gaussian_smoothing')
        #print(cfg['priorinfo']['emulate_tavtigian'])

        self.emulate_tavtigian = cfg['priorinfo'].getboolean('emulate_tavtigian')
        self.emulate_pejaver = cfg['priorinfo'].getboolean('emulate_pejaver')
        
        #print(self.emulate_tavtigian)
        if not (self.emulate_tavtigian or self.emulate_pejaver):
            self.alpha = float(cfg['priorinfo']['alpha'])
