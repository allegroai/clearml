from collections import namedtuple
from functools import partial

import six
from pathlib2 import Path

from ..backend_api.services import models
from .base import IdObjectBase
from .util import make_message
from ..storage import StorageHelper
from ..utilities.async_manager import AsyncManagerMixin

ModelPackage = namedtuple('ModelPackage', 'weights design')


class ModelDoesNotExistError(Exception):
    pass


class _StorageUriMixin(object):
    @property
    def upload_storage_uri(self):
        """ A URI into which models are uploaded """
        return self._upload_storage_uri

    @upload_storage_uri.setter
    def upload_storage_uri(self, value):
        self._upload_storage_uri = value.rstrip('/') if value else None


def create_dummy_model(upload_storage_uri=None, *args, **kwargs):
    class DummyModel(models.Model, _StorageUriMixin):
        def __init__(self, upload_storage_uri=None, *args, **kwargs):
            super(DummyModel, self).__init__(*args, **kwargs)
            self.upload_storage_uri = upload_storage_uri

        def update(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    return DummyModel(upload_storage_uri=upload_storage_uri, *args, **kwargs)


class Model(IdObjectBase, AsyncManagerMixin, _StorageUriMixin):
    """ Manager for backend model objects """

    _EMPTY_MODEL_ID = 'empty'

    @property
    def model_id(self):
        return self.id

    @property
    def storage(self):
        return StorageHelper.get(self.upload_storage_uri)

    def __init__(self, upload_storage_uri, cache_dir, model_id=None,
                 upload_storage_suffix='models', session=None, log=None):
        super(Model, self).__init__(id=model_id, session=session, log=log)
        self._upload_storage_suffix = upload_storage_suffix
        if model_id == self._EMPTY_MODEL_ID:
            # Set an empty data object
            self._data = models.Model()
        else:
            self._data = None
        self._cache_dir = cache_dir
        self.upload_storage_uri = upload_storage_uri

    def publish(self):
        self.send(models.SetReadyRequest(model=self.id, publish_task=False))
        self.reload()

    def _reload(self):
        """ Reload the model object """
        if self.id == self._EMPTY_MODEL_ID:
            return
        res = self.send(models.GetByIdRequest(model=self.id))
        return res.response.model

    def _upload_model(self, model_file, async_enable=False, target_filename=None, cb=None):
        if not self.upload_storage_uri:
            raise ValueError('Model has no storage URI defined (nowhere to upload to)')
        helper = self.storage
        target_filename = target_filename or Path(model_file).name
        dest_path = '/'.join((self.upload_storage_uri, self._upload_storage_suffix or '.', target_filename))
        result = helper.upload(
            src_path=model_file,
            dest_path=dest_path,
            async_enable=async_enable,
            cb=partial(self._upload_callback, cb=cb),
        )
        if async_enable:
            def msg(num_results):
                self.log.info("Waiting for previous model to upload (%d pending, %s)" % (num_results, dest_path))

            self._add_async_result(result, wait_on_max_results=2, wait_cb=msg)
        return dest_path

    def _upload_callback(self, res, cb=None):
        if res is None:
            self.log.debug('Starting model upload')
        elif res is False:
            self.log.info('Failed model upload')
        else:
            self.log.info('Completed model upload to %s' % res)
        if cb:
            cb(res)

    @staticmethod
    def _wrap_design(design):
        """
        Wrap design text with a dictionary.

        In the backend, the design is a dictionary with a 'design' key in it.
        For the client, it is a text. This function wraps a design string with
        the proper dictionary.

        :param design: If it is a dictionary, it mast have a 'design' key in it.
            In that case, return design as-is.
            If it is a string, return the dictionary {'design': design}.
            If it is None (or any False value), return the dictionary {'design': ''}

        :return: A proper design dictionary according to design parameter.
        """
        if isinstance(design, dict):
            if 'design' not in design:
                raise ValueError('design dictionary must have \'design\' key in it')

            return design

        return {'design': design if design else ''}

    @staticmethod
    def _unwrap_design(design):
        """
        Unwrap design text from a dictionary.

        In the backend, the design is a dictionary with a 'design' key in it.
        For the client, it is a text. This function unwraps a design string from
        the dictionary.

        :param design: If it is a dictionary with a 'design' key in it, return
            design['design'].
            If it is a dictionary without 'design' key, return the first value
            in it's values list.
            If it is an empty dictionary, None, or any other False value,
            return an empty string.
            If it is a string, return design as-is.

        :return: The design string according to design parameter.
        """
        if not design:
            return ''

        if isinstance(design, six.string_types):
            return design

        if isinstance(design, dict):
            if 'design' in design:
                return design['design']

            return list(design.values())[0]

        raise ValueError('design must be a string or a dictionary with at least one value')

    def update(self, model_file=None, design=None, labels=None, name=None, comment=None, tags=None,
               task_id=None, project_id=None, parent_id=None, uri=None, framework=None,
               upload_storage_uri=None, target_filename=None, iteration=None):
        """ Update model weights file and various model properties """

        if self.id is None:
            if upload_storage_uri:
                self.upload_storage_uri = upload_storage_uri
            self._create_empty_model(self.upload_storage_uri)

        # upload model file if needed and get uri
        uri = uri or (self._upload_model(model_file, target_filename=target_filename) if model_file else self.data.uri)
        # update fields
        design = self._wrap_design(design) if design else self.data.design
        name = name or self.data.name
        comment = comment or self.data.comment
        tags = tags or self.data.tags
        labels = labels or self.data.labels
        task = task_id or self.data.task
        project = project_id or self.data.project
        parent = parent_id or self.data.parent

        self.send(models.EditRequest(
            model=self.id,
            uri=uri,
            name=name,
            comment=comment,
            tags=tags,
            labels=labels,
            design=design,
            task=task,
            project=project,
            parent=parent,
            framework=framework or self.data.framework,
            iteration=iteration,
        ))
        self.reload()

    def update_and_upload(self, model_file, design=None, labels=None, name=None, comment=None,
                          tags=None, task_id=None, project_id=None, parent_id=None, framework=None, async_enable=False,
                          target_filename=None, cb=None, iteration=None):
        """ Update the given model for a given task ID """
        if async_enable:
            def callback(uploaded_uri):
                if uploaded_uri is None:
                    return

                # If not successful, mark model as failed_uploading
                if uploaded_uri is False:
                    uploaded_uri = '{}/failed_uploading'.format(self._upload_storage_uri)

                self.update(
                    uri=uploaded_uri,
                    task_id=task_id,
                    name=name,
                    comment=comment,
                    tags=tags,
                    design=design,
                    labels=labels,
                    project_id=project_id,
                    parent_id=parent_id,
                    framework=framework,
                    iteration=iteration,
                )

                if cb:
                    cb(model_file)

            uri = self._upload_model(model_file, async_enable=async_enable, target_filename=target_filename, cb=callback)
            return uri
        else:
            uri = self._upload_model(model_file, async_enable=async_enable, target_filename=target_filename)
            self.update(
                uri=uri,
                task_id=task_id,
                name=name,
                comment=comment,
                tags=tags,
                design=design,
                labels=labels,
                project_id=project_id,
                parent_id=parent_id,
                framework=framework,
            )

            return uri

    def _complete_update_for_task(self, uri, task_id=None, name=None, comment=None, tags=None, override_model_id=None,
                                  cb=None):
        if self._data:
            name = name or self.data.name
            comment = comment or self.data.comment
            tags = tags or self.data.tags
            uri = (uri or self.data.uri) if not override_model_id else None

        res = self.send(
            models.UpdateForTaskRequest(task=task_id, uri=uri, name=name, comment=comment, tags=tags,
                                        override_model_id=override_model_id))
        if self.id is None:
            # update the model id. in case it was just created, this will trigger a reload of the model object
            self.id = res.response.id
        else:
            self.reload()
        try:
            if cb:
                cb(uri)
        except Exception as ex:
            self.log.warning('Failed calling callback on complete_update_for_task: %s' % str(ex))
            pass

    def update_for_task_and_upload(
            self, model_file, task_id, name=None, comment=None, tags=None, override_model_id=None, target_filename=None,
            async_enable=False, cb=None, iteration=None):
        """ Update the given model for a given task ID """
        if async_enable:
            callback = partial(
                self._complete_update_for_task, task_id=task_id, name=name, comment=comment, tags=tags,
                override_model_id=override_model_id, cb=cb)
            uri = self._upload_model(model_file, target_filename=target_filename, async_enable=async_enable, cb=callback)
            return uri
        else:
            uri = self._upload_model(model_file, target_filename=target_filename, async_enable=async_enable)
            self._complete_update_for_task(uri, task_id, name, comment, tags, override_model_id)
            _ = self.send(models.UpdateForTaskRequest(task=task_id, uri=uri, name=name, comment=comment, tags=tags,
                                                      override_model_id=override_model_id, iteration=iteration))
            return uri

    def update_for_task(self, task_id, uri=None, name=None, comment=None, tags=None, override_model_id=None):
        self._complete_update_for_task(uri, task_id, name, comment, tags, override_model_id)

    @property
    def model_design(self):
        """ Get the model design. For now, this is stored as a single key in the design dict. """
        try:
            return self._unwrap_design(self.data.design)
        except ValueError:
            # no design is yet specified
            return None

    @property
    def labels(self):
        try:
            return self.data.labels
        except ValueError:
            # no labels is yet specified
            return None

    @property
    def name(self):
        try:
            return self.data.name
        except ValueError:
            # no name is yet specified
            return None

    @property
    def comment(self):
        try:
            return self.data.comment
        except ValueError:
            # no comment is yet specified
            return None

    @property
    def tags(self):
        return self.data.tags

    @property
    def locked(self):
        if self.id is None:
            return False
        return bool(self.data.ready)

    def download_model_weights(self):
        """ Download the model weights into a local file in our cache """
        uri = self.data.uri
        helper = StorageHelper.get(uri, logger=self._log, verbose=True)
        return helper.download_to_file(uri, force_cache=True)

    @property
    def cache_dir(self):
        return self._cache_dir

    def save_model_design_file(self):
        """ Download model description file into a local file in our cache_dir """
        design = self.model_design
        filename = self.data.name + '.txt'
        p = Path(self.cache_dir) / filename
        # we always write the original model design to file, to prevent any mishaps
        # if p.is_file():
        #     return str(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(six.text_type(design))
        return str(p)

    def get_model_package(self):
        """ Get a named tuple containing the model's weights and design """
        return ModelPackage(weights=self.download_model_weights(), design=self.save_model_design_file())

    def get_model_design(self):
        """ Get model description (text) """
        return self.model_design

    @classmethod
    def get_all(cls, session, log=None, **kwargs):
        req = models.GetAllRequest(**kwargs)
        res = cls._send(session=session, req=req, log=log)
        return res

    def clone(self, name, comment=None, child=True, tags=None, task=None, ready=True):
        """
        Clone this model into a new model.
        :param name: Name for the new model
        :param comment: Optional comment for the new model
        :param child: Should the new model be a child of this model? (default True)
        :return: The new model's ID
        """
        data = self.data
        assert isinstance(data, models.Model)
        parent = self.id if child else None
        req = models.CreateRequest(
            uri=data.uri,
            name=name,
            labels=data.labels,
            comment=comment or data.comment,
            tags=tags or data.tags,
            framework=data.framework,
            design=data.design,
            ready=ready,
            project=data.project,
            parent=parent,
            task=task,
        )
        res = self.send(req)
        return res.response.id

    def _create_empty_model(self, upload_storage_uri=None):
        upload_storage_uri = upload_storage_uri or self.upload_storage_uri
        name = make_message('Anonymous model %(time)s')
        uri = '{}/uploading_file'.format(upload_storage_uri or 'file://')
        req = models.CreateRequest(uri=uri, name=name, labels={})
        res = self.send(req)
        if not res:
            return False
        self.id = res.response.id
        return True
