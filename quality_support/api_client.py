import urllib
import requests as requests


class ApiClient:
    def __init__(self, endpoint, project_id: str, api_key):
        self.endpoint = endpoint
        self.project_id = project_id
        self.api_key = api_key

    def _build_url(self, *args, **kwargs):
        return (
            "/".join([self.endpoint] + list(args))
            + "?"
            + urllib.parse.urlencode(kwargs)
        )

    def _get_request(self, *args, **kwargs):
        url = self._build_url(*args, **kwargs)
        response = requests.get(url, verify=False)
        if response.status_code < 300:
            return response.json()
        else:
            return {"error": response.content, "status_code": response.status_code}

    # === COMPUTE STATE === #
    def get_state(self):
        state = self._get_request("state")
        return state

    # === EMBEDDING ENDPOINTS === #
    def get_embedding_urls(self):
        urls = self._get_request(self.project_id, "embedding", "urls")
        return urls

    def start_embedding(
        self,
        filter_classes: Optional[str] = None,
        train_size: Optional[float] = None,
    ):
        optional_params = {}
        if filter_classes is not None:
            optional_params["filter_classes"] = filter_classes
        if train_size is not None:
            optional_params["train_size"] = train_size

        result = self._get_request(
            self.project_id,
            "embedding",
            "start",
            api_key=self.api_key,
            **optional_params
        )
        return result

    def get_embeddings_exist(self):
        return self._get_request(self.project_id, "embedding", "exists")

    # === TRAINING ENDPOINTS === #
    def start_training(self):
        embeddings_exist = self.get_embeddings_exist()
        if not all(embeddings_exist.values()):
            return {"error": "embeddings does not exist."}

        state = self._get_request(self.project_id, "training", "start")
        return state

    def get_training_models_exist(self):
        return self._get_request(self.project_id, "training", "exists")

    # === INFERENCE ENDPOINTS === #
    def start_inference(self):
        if not self.get_training_models_exist()["model_params_exist"]:
            return {"error": "Predictive models does not exist."}

        return self._get_request(self.project_id, "inference", "start")

    def get_inference_urls(self):
        return self._get_request(self.project_id, "inference", "urls")
