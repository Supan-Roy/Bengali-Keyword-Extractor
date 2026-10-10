import pandas as pd

def load_and_clean_dataset(dataset_path):
    df = pd.read_csv(dataset_path)

    df_clean = df.copy()
    df_clean["content"] = (
        df_clean["content"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove empty articles
    df_clean = df_clean[df_clean["content"] != ""].copy()

    # Remove duplicate URLs
    df_clean = df_clean.drop_duplicates(
        subset="url",
        keep="first"
    ).copy()

    # Remove metadata mistakenly stored as article content
    metadata_only = df_clean["content"].isin({"আল মাহফুজ"})
    df_clean = df_clean.loc[~metadata_only].copy()

    return df_clean