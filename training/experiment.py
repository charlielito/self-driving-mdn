import os
from pathlib import Path

import dataget
import dicto
import numpy as np
import pandas as pd
import tensorflow as tf
import typer
from tensorflow_probability import distributions as tfd

from . import estimator

# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def slice_parameter_vectors(parameter_vector, components, no_parameters=3):
    """ Returns an unpacked list of paramter vectors.
    """
    return [
        parameter_vector[:, i * components : (i + 1) * components]
        for i in range(no_parameters)
    ]


def main(
    params_path: Path = Path("training/params.yml"),
    cache: bool = False,
    viz: bool = False,
    debug: bool = False,
):
    if debug:
        import debugpy

        print("Waiting debuger....")
        debugpy.listen(("localhost", 5678))
        debugpy.wait_for_client()

    params = dicto.load(params_path)

    train_cache_path = Path("cache") / "train.feather"
    test_cache_path = Path("cache") / "test.feather"

    if cache and train_cache_path.exists() and test_cache_path.exists():
        print("Using cache...")

        df_train = pd.read_feather(train_cache_path)
        df_test = pd.read_feather(test_cache_path)

    else:
        # df = dataget.image.udacity_simulator().get()
        header = ["center", "left", "right", "steering", "throttle", "break", "speed"]
        df = pd.read_csv(
            os.path.join(params["dataset"], "driving_log.csv"), names=header
        )

        df_train, df_test = estimator.split(df, params)

        df_train = estimator.preprocess(df_train, params, "train", params["dataset"])
        df_test = estimator.preprocess(df_test, params, "test", params["dataset"])
        # df_train = estimator.preprocess(df_train, params, "train")
        # df_test = estimator.preprocess(df_test, params, "test")

        # cache data
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        train_cache_path.parent.mkdir(exist_ok=True)

        df_train.to_feather(train_cache_path)
        df_test.to_feather(test_cache_path)

    ds_train = estimator.get_dataset(df_train, params, "train")
    ds_test = estimator.get_dataset(df_test, params, "test")

    # Visualize dataset for debuggings
    if viz:
        import matplotlib.pyplot as plt

        iteraror = iter(ds_train)
        image_batch, steer_batch, weights = next(iteraror)
        for image, steer, weight in zip(image_batch, steer_batch, weights):
            plt.imshow(image.numpy())
            plt.title(f"Steering angle: {steer} weight {weight}")
            plt.show()

        return

    components = params["components"]

    def gnll_loss(y, parameter_vector):
        """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
        """
        alpha, mu, sigma = slice_parameter_vectors(
            parameter_vector, components
        )  # Unpack parameter vectors

        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alpha),
            components_distribution=tfd.Normal(loc=mu, scale=sigma),
        )

        log_likelihood = gm.log_prob(tf.transpose(y))  # Evaluate log-probability of y

        return -tf.reduce_mean(log_likelihood, axis=-1)

    model = estimator.get_model(params, components)
    # model = estimator.get_simclr(params)
    loss = gnll_loss if components is not None else "mse"
    metrics = None if components is not None else ["mae"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(params.lr), loss=loss, metrics=metrics,
    )

    model.summary()
    # exit()

    model.fit(
        ds_train,
        epochs=params.epochs,
        steps_per_epoch=params.steps_per_epoch,
        validation_data=ds_test,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=str(Path("summaries") / Path(model.name)), profile_batch=0
            )
        ],
    )

    # Export to saved model
    save_path = os.path.join("models", model.name)
    model.save(save_path)

    # Save also yml with configs
    dicto.dump(params, os.path.join(save_path, "params.yml"))
    # print(f"{save_path=}")


if __name__ == "__main__":
    typer.run(main)
