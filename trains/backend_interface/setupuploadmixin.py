from abc import abstractproperty

from ..backend_config.bucket_config import S3BucketConfig
from ..storage.helper import StorageHelper


class SetupUploadMixin(object):
    log = abstractproperty()
    storage_uri = abstractproperty()

    def setup_upload(
            self, bucket_name, host=None, access_key=None, secret_key=None, region=None, multipart=True, https=True, verify=True):
        """
        Setup upload options (currently only S3 is supported)

        :param bucket_name: AWS bucket name
        :type bucket_name: str
        :param host: Hostname. Only required in case a Non-AWS S3 solution such as a local Minio server is used)
        :type host: str
        :param access_key: AWS access key. If not provided, we'll attempt to obtain the key from the
            configuration file (bucket-specific, than global)
        :type access_key: str
        :param secret_key: AWS secret key. If not provided, we'll attempt to obtain the secret from the
            configuration file (bucket-specific, than global)
        :type secret_key: str
        :param multipart: Server supports multipart. Only required when using a Non-AWS S3 solution that doesn't support
            multipart.
        :type multipart: bool
        :param https: Server supports HTTPS. Only required when using a Non-AWS S3 solution that only supports HTTPS.
        :type https: bool
        :param region: Bucket region. Required if the bucket doesn't reside in the default region (us-east-1)
        :type region: str
        :param verify: Whether or not to verify SSL certificates. Only required when using a Non-AWS S3 solution that only supports HTTPS with self-signed certificate.
        :type verify: bool
        """
        self._bucket_config = S3BucketConfig(
            bucket=bucket_name,
            host=host,
            key=access_key,
            secret=secret_key,
            multipart=multipart,
            secure=https,
            region=region,
            verify=verify
        )
        self.storage_uri = ('s3://%(host)s/%(bucket_name)s' if host else 's3://%(bucket_name)s') % locals()
        StorageHelper.add_configuration(self._bucket_config, log=self.log)
