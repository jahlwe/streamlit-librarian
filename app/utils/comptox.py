# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 22:51:00 2026

@author: Jakob
"""

import ctxpy as ctx
# ctx-python looks like the thing to be working with

# apparently you can enter your api key in an .env file that is located
# somewhere --- great if we can get it in some place like that and have it
# be transferred with the docker image
api = "0c5e2d7b-b651-4a21-a798-98d853dc9859"

# there are different classes that reach different databases
chem = ctx.Chemical(x_api_key=api)




# OK. Lets say that we query a compound and we DON't have the
# dtxsid. Maybe we can first do the pcq, and maybe GET the
# dtxsid. if we dont? what can we do?

# We can get dtxsid like this. Or cas. 
res = chem.search(by='equals', query='Chlorpheniramine')[0]
chem.search(by='equals', query='132-22-9')
# maybe INCHIKEY!!! is what we want to use.
chem.search(by='equals', query='VKYKSIONXSXAKP-UHFFFAOYSA-N')
# Let's just use what we have and do what we can
# If we have DTXSID (maybe even give an option to enter that in pcq),
# we use that. If we dont, try CAS. If we dont have CAS, try name.
# Get DTXSID. Then move on to query the stuff people want.
# Maybe good to confirm by smiles (provided they are equivalent)
# if we do a name query.


# we will need some structure like this
query_specs = {
    'chem':['pkaaOperaPred','octanolWaterPartition','octanolAirPartitionCoeff'],
    'expo':[]
}




# this has some good predicted values for stuff
# does it always come formatted the same way?
res = chem.details(by='dtxsid', query='DTXSID7020182')
items = [i for i in res.items()]
names = [name for (name, data) in items]

res

res = chem.details(by='dtxsid', query='DTXSID0022804')
items_b = [i for i in res.items()]
names_b = [name for (name, data) in items] # looks like it

res = chem.details(by='dtxsid', query='DTXSID1022556')
items_c = [i for i in res.items()]
names_c = [name for (name, data) in items] # yes.

# What could be "extractables" from here?
# waterSolubilityTest, waterSolubilityOpera,
# viscosityCpCpTestPred, vaporPressureMmhqTestPred,
# vaporPressureMmhgOperaPred, soilAdsorptionCoefficient,
# thermalConductivity, oralRatLd50Mol, 
# octanolWaterPartition

# can we query through names as well?
chem.details(by='dtxsid', query='Fluvoxamine') # No

### According to github, chem only has search, details and msready. 
# search we will use, and details looks most relevant for us.










# what about expo

# Lots of stuff, kind of data heavy
# Maybe dont implement anything of this stuff yet

expo = ctx.Exposure(x_api_key=api)

# Readme says _qsur but its _qsurs
# Quantitative Structure_USE_Relationship
expo.search_qsurs(dtxsid='DTXSID2022628') 

# Multimedia Monitoring Database
expo.search_mmdb(by='dtxsid', query='DTXSID2022628')

# Exposure estimates
res = expo.search_exposures(by='pathways',dtxsid='DTXSID7020182')
res = expo.search_exposures(by='seem',dtxsid='DTXSID7020182')

# High-throughput toxicokinetics data
res = expo.search_httk(dtxsid='DTXSID2022628')








# OK what about hazard
haz = ctx.Hazard(x_api_key=api)

# ToxValDB
# cancer data
res = haz.search_toxvaldb(by='cancer', dtxsid='DTXSID1022556')
# 'human' didnt work as argument to 'by'.
res = haz.search_toxvaldb(by='all', dtxsid='DTXSID1022556')
res = haz.search_toxvaldb(by='genetox', dtxsid='DTXSID1022556')

# ToxRefDB, don't know what this is
res = haz.search_toxrefdb(by='dtxsid', query='DTXSID2023224', domain='effects')

# these don't always return data
haz.search_pprtv('DTXSID2023224')
haz.search_hawc('DTXSID2023224')
haz.search_iris('DTXSID2023224')
haz.search_adme_ivive('DTXSID2023224')

