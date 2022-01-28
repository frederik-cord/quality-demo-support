
## ApiClient
The ApiClient exposes the Api endpoints in a more pythonic way.
Use it as follows:

```python
from quality_support import ApiClient
api_client = ApiClient('<ipaddress>', '<project_id>', '<api_key>')

inference_urls = api_client.get_inference_urls()
```

You can get the ip from the Cord organization.


## Cord Data Grabber
This is a rather quick and dirty data grabber, which mixes functionality from the `cord.client` and the `Cord_pytorch_dataset` to retrieve data urls, object ontology, etc. 
The main usage intention is the following:
```python
# 1. Initialize the grabber:
project_id = '<your-project-id>'
api_key = '<associated-api-key>'
grabber = DataGrabber(project_id, api_key, cache_dir='./cord-cache')

# 2. Get item-wise meta data from the api
with open(root / 'label_results/metadata.json', 'r') as f:
    meta = json.load(f)

# 3. Get the image associated with whatever idx `i` in the metadata:
img: Path = grabber.image_from_hash(**meta[i])

# 4. Get the associated DataUnitObject for idx `i`
obj: DataUnitObject = grabber.object_from_hashes(**meta[i])

# .. do amazing things
``