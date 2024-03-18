
class HyperParameter:

    def __init__(
        self,
        name: str = 'n_estimators',
        value=3,
        limit=1,
        minimum=1,
        maximum=10,
        kind: 'type' = int
    ):
        self.name = name
        self.value = value
        self.test = value
        self.limit = limit
        self.min = minimum
        self.max = maximum
        self.kind = kind
