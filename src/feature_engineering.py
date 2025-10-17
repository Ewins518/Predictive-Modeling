
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def feature_engineering(df):
    df["house_age"] = df["yrsold"]-df["year_constructed"]
    df["house_age_when_remodeling"] = df["year_remod"] - df["year_constructed"]
    df["garage_age"] = df["yrsold"]-df["garageyrblt"]
    return df 


features_nom = [
    "sub_ms_class", "zoning_ms", "streetname", "alleyname", "contour_land", "configlot",
    "neighborhood", "c1", "c2", "type_building", "style_house", "roof_style", "roofmatl",
    "ext1", "ext2", "masvnrtype", "foundation", "heating", "garagetype", "miscfeature",
    "saletype", "salecondition"
]

ordered_levels = {
    "ovl_quality": list(range(1, 11)),
    "ovl_condition": list(range(1, 11)), 
    "exterqual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "extercond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "bsmtqual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "bsmtcond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "heatingqc": ["Po", "Fa", "TA", "Gd", "Ex"],
    "kitchenqual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "fireplacequ": ["Po", "Fa", "TA", "Gd", "Ex", "Nex"],
    "garagequal": ["Po", "Fa", "TA", "Gd", "Ex", "Nex"],
    "garagecond": ["Po", "Fa", "TA", "Gd", "Ex", "Nex"],
    "shape_lot": ["IR3", "IR2", "IR1", "Reg"],
    "slopeland": ["Sev", "Mod", "Gtl"],
    "bsmtexposure": ["No", "Mn", "Av", "Gd"],
    "bsmtfintype1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "bsmtfintype2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "garagefinish": ["Unf", "RFn", "Fin", "Nex"],
    "paveddrive": ["N", "P", "Y"],
    "util": ["NoSeWa", "NoSewr", "ELO", "AllPub"],
    "centralair": ["N", "Y"],
    "electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
}

def encode_categorical_features(df, features_nom=features_nom, ordered_levels=ordered_levels):

    df_encoded = df.copy()
    encoders = {'ordinal': {}, 'onehot': None}

    # prdinal encoding
    for col, categories in ordered_levels.items():
        if col in df_encoded.columns:
            oe = OrdinalEncoder(
                categories=[categories],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            df_encoded[[col]] = oe.fit_transform(df_encoded[[col]].fillna('NA'))
            encoders['ordinal'][col] = oe

    # one-Hot Encoding for Nominal
    nominal_cols = [col for col in features_nom if col in df_encoded.columns]

    if nominal_cols:
        ohe = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
            drop=None
        )
        onehot_encoded = ohe.fit_transform(df_encoded[nominal_cols])
        onehot_cols = ohe.get_feature_names_out(nominal_cols)

        df_onehot = pd.DataFrame(onehot_encoded, columns=onehot_cols, index=df_encoded.index)
        
        # drop original nominal columns and join encoded
        df_encoded = pd.concat([df_encoded.drop(columns=nominal_cols), df_onehot], axis=1)
        encoders['onehot'] = ohe

    return df_encoded, encoders
