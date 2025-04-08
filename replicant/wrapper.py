class MLFunctionWrapper:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def __call__(self, *args, **kwargs):
        # Convert positional + keyword args into ordered list
        input_row = self._assemble_input_row(args, kwargs)
        return self.model.predict([input_row])[0]  # single prediction

    def _assemble_input_row(self, args, kwargs):
        data = {}
        for i, name in enumerate(self.feature_names):
            if i < len(args):
                data[name] = args[i]
            elif name in kwargs:
                data[name] = kwargs[name]
            else:
                raise ValueError(f"Missing argument: {name}")
        return [data[name] for name in self.feature_names]
