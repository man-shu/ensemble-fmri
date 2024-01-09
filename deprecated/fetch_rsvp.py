import ibc_api.utils as fetcher

db = fetcher.get_info(data_type="preprocessed")
filtered_db = fetcher.filter_data(db, task_list=["RSVPLanguage"])
fetcher.authenticate()
downloaded_db = fetcher.download_data(filtered_db)
