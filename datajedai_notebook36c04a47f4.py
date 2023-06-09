#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import pandas as pd 
import seaborn as sns




# just to see where things are
get_ipython().system(' ls -la ../input')




def get_shape(df):
    return pd.DataFrame({'row':df.shape[0], 'features': df.shape[1]}, index=['count'])




df = pd.read_csv("../input/properties_2016.csv")




#get_shape(df)
pd.DataFrame( dict(zip( ('rows','dims'), df.shape ) ) , index=[1])




pd.options.display.float_format = '{:.2f}'.format
df.describe().T




# for col in df.columns[:2]:
#     sns.countplot('taxamount', data=df)
sns.distplot(df.taxamount)




# pivot the output to fit the scren
df.head().T




dftrn = pd.read_csv("../input/train_2016.csv")




get_shape(dftrn)




dftrn.head()




pdm = pd.merge( dftrn, df , how='left', on=['parcelid'])




pdm.head().T




pdm.columns




pdm.rename(columns={
    "parselid": "parsel_id",
    "logerror": "log_error",
    "transactiondate": "transaction_date",
    "airconditioningtypeid": "ac_type_id",
    "architecturalstyletypeid": "style_id",
    "decktypeid": "deck_type_id",

    "basementsqft": "basement_size",
    "finished"

    "bathroomcnt": "baths",
    "bedroomcnt": "rooms",
    "roomcnt" : "total_rooms",
    

    "calculatedbathnbr": "total_baths",
    "calculatedfinishedsquarfeet" : "cal_finished_size",
    

    "buildingclasstypeid": "building_class_type_id",
    "buildingqualitytypeid" : "building_quality_type_id",
    
    "finishedfloor1squarefeet": "finished_fl_1_size",
    "calculatedfinishedsquarefeet" : "total_living_area",
    
    "finishedsquarefeet12" : "finished_living_area_size",
    "finishedsquarefeet13" : "perimeter_living_area",
    "finishedsquarefeet15" : "total_area",
    
    "finishedsquarefeet50" : "finished_area_fl_1",
    
    "fireplacecnt" : "no_of_fireplaces",
    
    "fullbathcnt": "no_of_full_baths",
    "garagecarcnt" : "no_of_garages",
    "garagetotalsqft" : "total_garage_area",
    "hashottuborspa" : "spa_or_hot_tub_present",
    "heatingorsystemtypeid": "heating_system_type_id",
    "lotsizesquarefeet" : "lot_area",
    "poolcnt" : "no_of_pools",
    "poolsizesum" : "total_pool_area",
    
    "pooltypeid10": "spa_or_hot_tub",
    "pooltypeid2" : "pool_with_spa_or_hot_tub",
    "pooltypeid7" : "pool_without_spa_or_hot_tub",
    "propertycountylandusecode" : "county_zone_code",
    "propertylandusetypeid" : "county_zone_type_id",
    "propertyzoningdesc": "zone_desc",
    
    "rawcensustractandblock": "census_tract_block_id",
    "regionidcity" : "city_id",
    "regionidcounty" : "county_id",
    "regionidneighborhood" : "neighborhood_id",
    "regionidzip" : "zip_id",
    
    "storytypeid": "story_type_id",
    "typeconstructiontypeid" : "construction_material_type_id",
    
    "unitcnt": "no_of_units",
    
    "yardbuildingsqft17": "patio_area",
    "yardbuildingsqft26": "shed_area",
    
    "yearbuilt": "built_year",
    "numberofstories" : "no_of_stories",
    "fireplaceflag": "fireplace_present",
    "structuretaxvaluedollarcnt": "structure_tax_assessed",
    "landtaxvaluedollarcnt" : "land_tax_assessed",
    "taxvaluedollarcnt" : "total_tax_assessed",
    "assessmentyear": "assessment_year",
    "taxamount" : "tax",
    "taxdelinquencyflag": "tax_delinquency",
    "taxdelinquencyyear" : "tax_delinquency_year",
    "censustractandblock" : "census_tract_and_block"
    
}, inplace=True)




def show_counts_per_feature():
    max_len = pdm.parcelid.count()
    _df = pd.DataFrame( pdm.count() ,columns=['count'])            .sort_values(['count'])
    _df['missing_count'] = _df['count'].apply( lambda x : max_len - x)
    _df[ _df.missing_count > 0 ].plot(kind='barh',figsize=(10,20))
show_counts_per_feature()




pdm.tax_delinquency.unique()




pdm['tax_delinquency'].fillna('N', inplace=True)
pdm.rooms.fillna(0,inplace=True)
pdm.total_rooms.fillna(0, inplace=True)




show_counts_per_feature()




pdm.total_rooms.unique()





























































