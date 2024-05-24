from abc import ABC, abstractmethod
import io
import glob
import os
import pickle
import tempfile
from typing import Any, Dict, Union

import boto3
import tensorflow as tf
from pathlib import Path


def create_client(cfg: Dict[str, Any], service: str = "s3"):
    """Create a boto3 client to interact with AWS services.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Configuration dictionary for the client.
    service : str
        Service name.

    Returns
    -------
    boto3.client
        Boto3 client.
    """
    boto3.setup_default_session(**cfg["session"])
    return boto3.client(service)


class StorageUtil(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    def store_obj(self, obj: Any, file_path: Union[str, Path]):
        pass

    @abstractmethod
    def load_obj(self, file_path: Union[str, Path]):
        pass

    @abstractmethod
    def list_dir(self, dir: Union[str, Path]):
        pass

    @abstractmethod
    def store_fig(self, fig, file_path: Union[str, Path]):
        pass

    @abstractmethod
    def store_policy(self, agent, file_path: Union[str, Path]):
        pass

    @abstractmethod
    def load_policy(self, agent, file_path: Union[str, Path]):
        pass


class CloudStorageUtil(StorageUtil):
    def __init__(self, config: Dict[str, Any]):
        """Initialize CloudStorageUtil with config and bucket.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration for the cloud storage.
        bucket : str
            S3 bucket name.
        service : str, optional
            Service name, defaults to "s3".
        """
        boto3.setup_default_session(**config["session"])
        self.s3_client = boto3.client("s3")
        self.bucket = config["bucket_name"]

    def store_obj(self, obj: Any, file_path: Union[str, Path]):
        """Upload the object as a pkl.gz file to the s3 bucket.

        Parameters
        ----------
        obj : Any
            Object to be uploaded.
        file_path : Union[str, Path]
            File path in the bucket.
        """
        file_path = str(file_path)
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=file_path,
            Body=pickle.dumps(obj),
        )

    def load_obj(self, file_path: Union[str, Path]):
        """Download the object from the s3 bucket and read the gzip file.

        Parameters
        ----------
        file_path : Union[str, Path]
            File path in the bucket.

        Returns
        -------
        Any
            Downloaded object.
        """
        file_path = str(file_path)
        response = self.s3_client.get_object(Bucket=self.bucket, Key=file_path)
        obj = pickle.loads(response["Body"].read())
        return obj

    def list_dir(self, dir: Union[str, Path]):
        """List all the objects in the directory from the s3 bucket.

        Parameters
        ----------
        dir : Union[str, Path]
            Directory in the bucket.

        Returns
        -------
        List[str]
            List of objects.
        """
        dir = str(dir)
        paginator = self.s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=dir)

        all_objects = []
        for page in page_iterator:
            if "Contents" in page:
                all_objects.extend(page["Contents"])
        all_objects = [obj["Key"] for obj in all_objects]

        return all_objects

    def store_fig(self, fig, file_path: Union[str, Path]):
        """Upload the figure as a png file to the s3 bucket.

        Parameters
        ----------
        fig : object
            Figure object.
        file_path : Union[str, Path]
            File path in the bucket.
        """
        file_path = str(file_path)
        img_data = io.BytesIO()
        fig.savefig(img_data, format="png")
        img_data.seek(0)
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=file_path,
            Body=img_data,
        )

    def store_policy(self, agent, file_path: Union[str, Path]):
        """Upload the policy as a checkpoint file to the s3 bucket.

        Parameters
        ----------
        agent : object
            Trained tf agent object.
        file_path : Union[str, Path]
            File path in the bucket.
        """
        file_path = str(file_path)
        with tempfile.TemporaryDirectory() as temp_dir_path:
            checkpoint_dir = str(temp_dir_path)
            checkpoint = tf.train.Checkpoint(agent=agent)
            checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
            checkpoint_manager.save()

            for file in glob.glob(checkpoint_dir + "/*", recursive=True):
                self.s3_client.upload_file(file, self.bucket, str(Path(file_path, "checkpoint", file.split("/")[-1])))

    def load_policy(self, agent, file_path: Union[str, Path]):
        """Download the policy from the s3 bucket and restore the agent.

        Parameters
        ----------
        agent : object
            Tf agent object without training.
        file_path : Union[str, Path]
            Directory in the bucket.

        Returns
        -------
        object
            Trained tf agent object.
        """
        file_path = str(file_path)
        with tempfile.TemporaryDirectory() as temp_dir_path:
            for file in self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=file_path,
            )["Contents"]:
                self.s3_client.download_file(self.bucket, file["Key"], str(Path(temp_dir_path, file["Key"].split("/")[-1])))

            latest_checkpoint = tf.train.latest_checkpoint(str(temp_dir_path))
            checkpoint = tf.train.Checkpoint(agent=agent)
            checkpoint.restore(latest_checkpoint)

            return checkpoint.agent


class LocalStorageUtil(StorageUtil):
    def __init__(self, *args, **kwargs):
        pass

    def store_obj(self, obj: Any, file_path: Union[str, Path]):
        """Store the object to a local directory as a pkl.gz file.

        Parameters
        ----------
        obj : Any
            Object to be stored.
        file_path : Union[str, Path]
            File path to store the object.
        """
        file_path = str(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_path = file_path + ".pkl" if not file_path.endswith(".pkl") else file_path
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

    def load_obj(self, file_path: Union[str, Path]):
        """Load the object from a local directory.

        Parameters
        ----------
        file_path : Union[str, Path]
            File path to load the object.

        Returns
        -------
        Any
            Loaded object.
        """
        file_path = str(file_path)
        file_path = file_path + ".pkl" if not file_path.endswith(".pkl") else file_path
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def list_dir(self, dir: Union[str, Path]):
        """List all files in a directory from local storage.

        Parameters
        ----------
        dir : Union[str, Path]
            Directory to list files from.

        Returns
        -------
        List[str]
            List of file paths.
        """
        dir = str(dir)
        dirs = glob.glob(dir + "*")
        paths = []
        for dir in dirs:
            for root, dirs, files in os.walk(dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    paths.append(file_path)
        return paths

    def store_fig(self, fig, file_path: Union[str, Path]):
        """Store the figure as a png file to the local directory.

        Parameters
        ----------
        fig : object
            Figure object.
        file_path : Union[str, Path]
            File path to store the figure.
        """
        file_path = str(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fig.savefig(file_path, dpi=fig.dpi)

    def store_policy(self, agent, file_path: Union[str, Path]):
        """Store the policy as a checkpoint file to the local directory.

        Parameters
        ----------
        agent : object
            Trained tf agent object.
        file_path : Union[str,Path]
            Directory to store the policy.
        """
        file_path = str(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        checkpoint = tf.train.Checkpoint(agent=agent)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, file_path, max_to_keep=1)
        checkpoint_manager.save()

    def load_policy(self, agent, file_path: Union[str, Path]):
        """Load the policy from the local directory and restore the agent.

        Parameters
        ----------
        agent : object
            Tf agent object without training.
        file_path : Union[str, Path]
            Directory to load the policy.

        Returns
        -------
        object
            Trained tf agent object.
        """
        file_path = str(file_path)
        latest_checkpoint = tf.train.latest_checkpoint(file_path)
        checkpoint = tf.train.Checkpoint(agent=agent)
        checkpoint.restore(latest_checkpoint)
        return checkpoint.agent
