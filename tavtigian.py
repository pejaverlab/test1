from scipy.optimize import fsolve

def evidence_to_plr(opvst, npsu, npm, npst, npvst):
    return opvst**( (npsu/8) + (npm/4) + (npst/2) + npvst )

def odds_to_postP(plr, priorP):
    postP = plr * priorP / ((plr - 1) * priorP + 1)
    return postP

def get_postP(c, prior, npsu, npm, npst, npvst):
    plr = evidence_to_plr(c, npsu, npm, npst, npvst)
    return odds_to_postP(plr, prior)

def get_postP_moderate(c, prior):
    return get_postP(c,prior,2,0,1,0) - 0.9

def get_tavtigian_c(prior):
    return fsolve(get_postP_moderate, 300 , args=(prior))
