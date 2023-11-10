import configparser

alpha = None
c = None
B = None
discountonesided = None
windowclinvarpoints = None
windowgnomadfraction = None

def load_config(filepath):

    cfg = configparser.ConfigParser()
    cfg.read(filepath)
    cfg.sections()

    global alpha
    global c
    global B
    global discountonesided
    global windowclinvarpoints
    global windowgnomadfraction


    alpha = float(cfg['tuningparameters']['alpha'])
    c = int(cfg['tuningparameters']['c'])
    B = int(cfg['tuningparameters']['B'])
    discountonesided = float(cfg['tuningparameters']['discountonesided'])
    windowclinvarpoints = int(cfg['tuningparameters']['windowclinvarpoints'])
    windowgnomadfraction = float(cfg['tuningparameters']['windowgnomadfraction'])
