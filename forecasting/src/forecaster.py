import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset

from uni2ts.model.moirai import (
    MoiraiForecast,
    MoiraiModule,
)


class MoiraiForecaster:

    def __init__(
        self,
        checkpoint,
        prediction_length=15,
        context_length=128,
        num_samples=100,
        batch_size=32,
        patch_size="auto",
    ):

        self.prediction_length = prediction_length

        if isinstance(patch_size, str) and patch_size.isdigit():
            patch_size = int(patch_size)

        print(f"Loading Moirai model: {checkpoint}")

        module = MoiraiModule.from_pretrained(checkpoint)

        self.model = MoiraiForecast(
            module=module,
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        self.predictor = self.model.create_predictor(
            batch_size=batch_size
        )

        print("Moirai model loaded successfully.")

    def predict(self, series):
        """
        `series`: 1-minute pandas Series with a DatetimeIndex.
        Returns dict of numpy arrays (mean / lower / upper), one value
        per forecast minute.
        """

        values = np.asarray(series.values, dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError("Series contains NaN or Inf values")

        start = series.index[0]
        if start.tzinfo is not None:
            start = start.tz_convert("UTC").tz_localize(None)

        dataset = ListDataset(
            [
                {
                    "start": pd.Period(start, freq="min"),
                    "target": values,
                }
            ],
            freq="1min",
        )

        forecast = list(self.predictor.predict(dataset))[0]

        samples = np.asarray(forecast.samples)

        # Shape: number of samples x forecast horizon.
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        return {
            "mean": np.mean(samples, axis=0),
            "lower": np.percentile(samples, 10, axis=0),
            "upper": np.percentile(samples, 90, axis=0),
        }
