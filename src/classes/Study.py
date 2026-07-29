class Study:
    def __init__(self, data, scenario, users):
        self._data = None
        self.data = data
        self._metadata = {"scenario": scenario, "users": users}

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        if not isinstance(value, dict):
            raise ValueError("Metadata needs to be a dictionary.")
        self._metadata = value