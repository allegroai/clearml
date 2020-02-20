import abc
import warnings
from copy import copy
from operator import itemgetter
from os import getenv

import furl
import six
from attr import attrib, attrs


def _none_to_empty_string(maybe_string):
    return maybe_string if maybe_string is not None else ""


def _url_stripper(bucket):
    bucket = _none_to_empty_string(bucket)
    bucket = bucket.strip("\"'").rstrip("/")
    return bucket


@attrs
class S3BucketConfig(object):
    bucket = attrib(type=str, converter=_url_stripper, default="")
    host = attrib(type=str, converter=_none_to_empty_string, default="")
    key = attrib(type=str, converter=_none_to_empty_string, default="")
    secret = attrib(type=str, converter=_none_to_empty_string, default="")
    multipart = attrib(type=bool, default=True)
    acl = attrib(type=str, converter=_none_to_empty_string, default="")
    secure = attrib(type=bool, default=True)
    region = attrib(type=str, converter=_none_to_empty_string, default="")

    def update(self, key, secret, multipart=True, region=None):
        self.key = key
        self.secret = secret
        self.multipart = multipart
        self.region = region

    def is_valid(self):
        return self.key and self.secret

    def get_bucket_host(self):
        return self.bucket, self.host

    @classmethod
    def from_list(cls, dict_list, log=None):
        if not isinstance(dict_list, (tuple, list)) or not all(
            isinstance(x, dict) for x in dict_list
        ):
            raise ValueError("Expecting a list of configurations dictionaries")
        configs = [cls(**entry) for entry in dict_list]
        valid_configs = [conf for conf in configs if conf.is_valid()]
        if log and len(valid_configs) < len(configs):
            log.warning(
                "Invalid bucket configurations detected for {}".format(
                    ", ".join(
                        "/".join((config.host, config.bucket))
                        for config in configs
                        if config not in valid_configs
                    )
                )
            )
        return configs


BucketConfig = S3BucketConfig


@six.add_metaclass(abc.ABCMeta)
class BaseBucketConfigurations(object):
    def __init__(self, buckets=None, *_, **__):
        self._buckets = buckets or []
        self._prefixes = None

    def _update_prefixes(self, refresh=True):
        if self._prefixes and not refresh:
            return
        prefixes = (
            (config, self._get_prefix_from_bucket_config(config))
            for config in self._buckets
        )
        self._prefixes = sorted(prefixes, key=itemgetter(1), reverse=True)

    @abc.abstractmethod
    def _get_prefix_from_bucket_config(self, config):
        pass


class S3BucketConfigurations(BaseBucketConfigurations):
    def __init__(
        self, buckets=None, default_key="", default_secret="", default_region=""
    ):
        super(S3BucketConfigurations, self).__init__()
        self._buckets = buckets if buckets else list()
        self._default_key = default_key
        self._default_secret = default_secret
        self._default_region = default_region
        self._default_multipart = True

    @classmethod
    def from_config(cls, s3_configuration):
        config_list = S3BucketConfig.from_list(
            s3_configuration.get("credentials", [])
        )

        default_key = s3_configuration.get("key") or getenv("AWS_ACCESS_KEY_ID", "")
        default_secret = s3_configuration.get("secret") or getenv("AWS_SECRET_ACCESS_KEY", "")
        default_region = s3_configuration.get("region") or getenv("AWS_DEFAULT_REGION", "")

        default_key = _none_to_empty_string(default_key)
        default_secret = _none_to_empty_string(default_secret)
        default_region = _none_to_empty_string(default_region)

        return cls(config_list, default_key, default_secret, default_region)

    def add_config(self, bucket_config):
        self._buckets.insert(0, bucket_config)
        self._prefixes = None

    def remove_config(self, bucket_config):
        self._buckets.remove(bucket_config)
        self._prefixes = None

    def get_config_by_bucket(self, bucket, host=None):
        try:
            return next(
                bucket_config
                for bucket_config in self._buckets
                if (bucket, host) == bucket_config.get_bucket_host()
            )
        except StopIteration:
            pass

        return None

    def update_config_with_defaults(self, bucket_config):
        bucket_config.update(
            key=self._default_key,
            secret=self._default_secret,
            region=bucket_config.region or self._default_region,
            multipart=bucket_config.multipart or self._default_multipart,
        )

    def _get_prefix_from_bucket_config(self, config):
        scheme = "s3"
        prefix = furl.furl()

        if config.host:
            prefix.set(
                scheme=scheme,
                netloc=config.host.lower(),
                path=config.bucket.lower() if config.bucket else "",
            )
        else:
            prefix.set(scheme=scheme, path=config.bucket.lower())
            bucket = prefix.path.segments[0]
            prefix.path.segments.pop(0)
            prefix.set(netloc=bucket)

        return str(prefix)

    def get_config_by_uri(self, uri):
        """
        Get the credentials for an AWS S3 bucket from the config
        :param uri: URI of bucket, directory or file
        :return: bucket config
        :rtype: S3BucketConfig
        """

        def find_match(uri):
            self._update_prefixes(refresh=False)
            uri = uri.lower()
            res = (
                config
                for config, prefix in self._prefixes
                if prefix is not None and uri.startswith(prefix)
            )

            try:
                return next(res)
            except StopIteration:
                return None

        match = find_match(uri)

        if match:
            return match

        parsed = furl.furl(uri)

        if parsed.port:
            host = parsed.netloc
            parts = parsed.path.segments
            bucket = parts[0] if parts else None
        else:
            host = None
            bucket = parsed.netloc

        return S3BucketConfig(
            key=self._default_key,
            secret=self._default_secret,
            region=self._default_region,
            multipart=True,
            bucket=bucket,
            host=host,
        )


BucketConfigurations = S3BucketConfigurations


@attrs
class GSBucketConfig(object):
    bucket = attrib(type=str)
    subdir = attrib(type=str, converter=_url_stripper, default="")
    project = attrib(type=str, default=None)
    credentials_json = attrib(type=str, default=None)

    def update(self, **kwargs):
        for item in kwargs:
            if not hasattr(self, item):
                warnings.warn("Unexpected argument {} for update. Ignored".format(item))
            else:
                setattr(self, item, kwargs[item])


class GSBucketConfigurations(BaseBucketConfigurations):
    def __init__(self, buckets=None, default_project=None, default_credentials=None):
        super(GSBucketConfigurations, self).__init__(buckets)
        self._default_project = default_project
        self._default_credentials = default_credentials

        self._update_prefixes()

    @classmethod
    def from_config(cls, gs_configuration):
        default_credentials = getenv("GOOGLE_APPLICATION_CREDENTIALS") or {}

        if not gs_configuration:
            return cls(default_credentials=default_credentials)

        config_list = gs_configuration.get("credentials", [])
        buckets_configs = [GSBucketConfig(**entry) for entry in config_list]

        default_project = gs_configuration.get("project") or {}
        default_credentials = gs_configuration.get("credentials_json") or default_credentials

        return cls(buckets_configs, default_project, default_credentials)

    def add_config(self, bucket_config):
        self._buckets.insert(0, bucket_config)
        self._update_prefixes()

    def remove_config(self, bucket_config):
        self._buckets.remove(bucket_config)
        self._update_prefixes()

    def update_config_with_defaults(self, bucket_config):
        bucket_config.update(
            project=bucket_config.project or self._default_project,
            credentials_json=bucket_config.credentials_json
            or self._default_credentials,
        )

    def get_config_by_uri(self, uri):
        """
        Get the credentials for a Google Storage bucket from the config
        :param uri: URI of bucket, directory or file
        :return: bucket config
        :rtype: GSBucketConfig
        """

        res = (
            config
            for config, prefix in self._prefixes
            if prefix is not None and uri.lower().startswith(prefix)
        )

        try:
            return next(res)
        except StopIteration:
            pass

        parsed = furl.furl(uri)

        return GSBucketConfig(
            bucket=parsed.netloc,
            subdir=str(parsed.path),
            project=self._default_project,
            credentials_json=self._default_credentials,
        )

    def _get_prefix_from_bucket_config(self, config):
        prefix = furl.furl(scheme="gs", netloc=config.bucket, path=config.subdir)
        return str(prefix)


@attrs
class AzureContainerConfig(object):
    account_name = attrib(type=str)
    account_key = attrib(type=str)
    container_name = attrib(type=str, default=None)


class AzureContainerConfigurations(object):
    def __init__(self, container_configs=None):
        super(AzureContainerConfigurations, self).__init__()
        self._container_configs = container_configs or []

    @classmethod
    def from_config(cls, configuration):
        default_account = getenv("AZURE_STORAGE_ACCOUNT")
        default_key = getenv("AZURE_STORAGE_KEY")

        default_container_configs = []
        if default_account and default_key:
            default_container_configs.append(AzureContainerConfig(
                account_name=default_account, account_key=default_key
            ))

        if configuration is None:
            return cls(default_container_configs)

        containers = configuration.get("containers", list())
        container_configs = [AzureContainerConfig(**entry) for entry in containers] + default_container_configs

        return cls(container_configs)

    def get_config_by_uri(self, uri):
        """
        Get the credentials for an Azure Blob Storage container from the config
        :param uri: URI of container or blob
        :return: container config
        :rtype: AzureContainerConfig
        """
        f = furl.furl(uri)
        account_name = f.host.partition(".")[0]

        if not f.path.segments:
            raise ValueError(
                "URI {} is missing a container name (expected "
                "[https/azure]://<account-name>.../<container-name>)".format(
                    uri
                )
            )

        container = f.path.segments[0]

        config = copy(self.get_config(account_name, container))

        if config and not config.container_name:
            config.container_name = container

        return config

    def get_config(self, account_name, container):
        return next(
            (
                config
                for config in self._container_configs
                if config.account_name == account_name and (
                    not config.container_name
                    or config.container_name == container
                )
            ),
            None
        )
