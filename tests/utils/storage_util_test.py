import pytest
from unittest.mock import Mock
import boto3
from pathlib import Path
import pickle
import os
from tf_agents.agents.random.random_agent import RandomAgent
from retail_agent.utils.storage_util import CloudStorageUtil, LocalStorageUtil
from retail_agent.utils.train_eval_util import initialize_agent

@pytest.mark.skip(reason="need aws sso credentials to run test")
class TestCloudStorageUtil:
    @pytest.fixture
    def cloud_storage_util(self, config):
        return CloudStorageUtil(config["cloud"])

    def test_store_obj(self, cloud_storage_util):
        obj = "test"
        file_path = "test.pkl"
        cloud_storage_util.store_obj(obj, file_path)
        s3_file = cloud_storage_util.list_dir(file_path)
        assert len(s3_file) == 1

    def test_load_obj(self, cloud_storage_util):
        obj = cloud_storage_util.load_obj("test.pkl")
        assert obj == "test"

    def test_list_dir(self, cloud_storage_util):
        s3_file = cloud_storage_util.list_dir('test.pkl')
        assert len(s3_file) == 1
        assert 'test.pkl' in s3_file

        cloud_storage_util.s3_client.delete_object(Bucket=cloud_storage_util.bucket, Key='test.pkl')

    def test_store_fig(self, cloud_storage_util, tmp_path):
        fig = Mock()
        file_path = str(tmp_path / "test.png")
        cloud_storage_util.store_fig(fig, file_path)
        s3_file = cloud_storage_util.list_dir(file_path)
        assert len(s3_file) == 1
        assert file_path in s3_file

        cloud_storage_util.s3_client.delete_object(Bucket=cloud_storage_util.bucket, Key=file_path)

    def test_store_policy(self, cloud_storage_util, config):
        agent = initialize_agent(config)
        file_path = "test_checkpoint"
        cloud_storage_util.store_policy(agent, file_path)
        s3_files = cloud_storage_util.list_dir(file_path)

        assert len(s3_files) == 3
        

    def test_load_policy(self, cloud_storage_util, config):
        agent = initialize_agent(config)
        file_path = "test_checkpoint/checkpoint"
        cloud_storage_util.load_policy(agent, file_path)
        assert isinstance(agent, RandomAgent)

        s3_files = cloud_storage_util.list_dir(file_path)
        for file in s3_files:
            cloud_storage_util.s3_client.delete_object(Bucket=cloud_storage_util.bucket, Key=file)

class TestLocalStorageUtil:
    @pytest.fixture
    def local_storage_util(self):
        return LocalStorageUtil()

    def test_store_obj(self, local_storage_util, tmp_path):
        obj = "test"
        file_path = tmp_path / "test.pkl"
        local_storage_util.store_obj(obj, str(file_path))
        assert file_path.exists()

    def test_load_obj(self, local_storage_util, tmp_path):
        obj = "test"
        file_path = tmp_path / "test.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        loaded_obj = local_storage_util.load_obj(str(file_path))
        assert loaded_obj == obj

    def test_list_dir(self, local_storage_util, tmp_path):
        dir_path = tmp_path / "test_dir"
        dir_path.mkdir()
        files = ["file1.pkl", "file2.pkl"]
        for file in files:
            (dir_path / file).touch()
        dir_contents = local_storage_util.list_dir(str(dir_path))
        assert len(dir_contents) == len(files)
        for file in files:
            assert str(dir_path / file) in dir_contents

    def test_store_fig(self, local_storage_util, tmp_path):
        fig = Mock()
        file_path = tmp_path / "test.png"
        local_storage_util.store_fig(fig, str(file_path))
        fig.savefig.assert_called_once_with(str(file_path), dpi=fig.dpi)

    def test_store_policy(self, local_storage_util, tmp_path, config):
        agent = initialize_agent(config)
        file_path = tmp_path / "test"
        local_storage_util.store_policy(agent, str(file_path))
        check_point_files = list(file_path.glob("*"))
        assert len(check_point_files) == 3


    def test_load_policy(self, local_storage_util, tmp_path, config):
        agent = initialize_agent(config)
        file_path = tmp_path / "test"
        local_storage_util.store_policy(agent, str(file_path))
        loaded_agent = local_storage_util.load_policy(agent, str(file_path))
