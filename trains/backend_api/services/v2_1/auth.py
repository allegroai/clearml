"""
auth service

This service provides authentication management and authorization
validation for the entire system.
"""
import six
import types
from datetime import datetime
import enum

from dateutil.parser import parse as parse_datetime

from ....backend_api.session import Request, BatchRequest, Response, DataModel, NonStrictDataModel, CompoundRequest, schema_property, StringEnum


class Credentials(NonStrictDataModel):
    """
    :param access_key: Credentials access key
    :type access_key: str
    :param secret_key: Credentials secret key
    :type secret_key: str
    """
    _schema = {
        'properties': {
            'access_key': {
                'description': 'Credentials access key',
                'type': ['string', 'null'],
            },
            'secret_key': {
                'description': 'Credentials secret key',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, access_key=None, secret_key=None, **kwargs):
        super(Credentials, self).__init__(**kwargs)
        self.access_key = access_key
        self.secret_key = secret_key

    @schema_property('access_key')
    def access_key(self):
        return self._property_access_key

    @access_key.setter
    def access_key(self, value):
        if value is None:
            self._property_access_key = None
            return
        
        self.assert_isinstance(value, "access_key", six.string_types)
        self._property_access_key = value

    @schema_property('secret_key')
    def secret_key(self):
        return self._property_secret_key

    @secret_key.setter
    def secret_key(self, value):
        if value is None:
            self._property_secret_key = None
            return
        
        self.assert_isinstance(value, "secret_key", six.string_types)
        self._property_secret_key = value


class CredentialKey(NonStrictDataModel):
    """
    :param access_key:
    :type access_key: str
    """
    _schema = {'properties': {'access_key': {'description': '', 'type': ['string', 'null']}}, 'type': 'object'}
    def __init__(
            self, access_key=None, **kwargs):
        super(CredentialKey, self).__init__(**kwargs)
        self.access_key = access_key

    @schema_property('access_key')
    def access_key(self):
        return self._property_access_key

    @access_key.setter
    def access_key(self, value):
        if value is None:
            self._property_access_key = None
            return
        
        self.assert_isinstance(value, "access_key", six.string_types)
        self._property_access_key = value


class AddUserRequest(Request):
    """
    Add a new user manually. Only supported in on-premises deployments

    :param secret_key: A secret key (used as the user's password)
    :type secret_key: str
    :param name: User name (makes the auth entry more readable)
    :type name: str
    :param company: Associated company ID. If not provided, the caller's company ID
        will be used
    :type company: str
    :param email: Email address uniquely identifying the user
    :type email: str
    :param provider: Provider ID indicating the external provider used to
        authenticate the user
    :type provider: str
    :param provider_user_id: Unique user ID assigned by the external provider
    :type provider_user_id: str
    :param provider_token: Provider-issued token for this user
    :type provider_token: str
    :param given_name: Given name
    :type given_name: str
    :param family_name: Family name
    :type family_name: str
    :param avatar: Avatar URL
    :type avatar: str
    """

    _service = "auth"
    _action = "add_user"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'avatar': {'description': 'Avatar URL', 'type': 'string'},
            'company': {
                'description': "Associated company ID. If not provided, the caller's company ID will be used",
                'type': 'string',
            },
            'email': {
                'description': 'Email address uniquely identifying the user',
                'type': 'string',
            },
            'family_name': {'description': 'Family name', 'type': 'string'},
            'given_name': {'description': 'Given name', 'type': 'string'},
            'name': {
                'description': 'User name (makes the auth entry more readable)',
                'type': 'string',
            },
            'provider': {
                'description': 'Provider ID indicating the external provider used to authenticate the user',
                'type': 'string',
            },
            'provider_token': {
                'description': 'Provider-issued token for this user',
                'type': 'string',
            },
            'provider_user_id': {
                'description': 'Unique user ID assigned by the external provider',
                'type': 'string',
            },
            'secret_key': {
                'description': "A secret key (used as the user's password)",
                'type': ['string', 'null'],
            },
        },
        'required': ['name', 'email'],
        'type': 'object',
    }
    def __init__(
            self, name, email, secret_key=None, company=None, provider=None, provider_user_id=None, provider_token=None, given_name=None, family_name=None, avatar=None, **kwargs):
        super(AddUserRequest, self).__init__(**kwargs)
        self.secret_key = secret_key
        self.name = name
        self.company = company
        self.email = email
        self.provider = provider
        self.provider_user_id = provider_user_id
        self.provider_token = provider_token
        self.given_name = given_name
        self.family_name = family_name
        self.avatar = avatar

    @schema_property('secret_key')
    def secret_key(self):
        return self._property_secret_key

    @secret_key.setter
    def secret_key(self, value):
        if value is None:
            self._property_secret_key = None
            return
        
        self.assert_isinstance(value, "secret_key", six.string_types)
        self._property_secret_key = value

    @schema_property('name')
    def name(self):
        return self._property_name

    @name.setter
    def name(self, value):
        if value is None:
            self._property_name = None
            return
        
        self.assert_isinstance(value, "name", six.string_types)
        self._property_name = value

    @schema_property('company')
    def company(self):
        return self._property_company

    @company.setter
    def company(self, value):
        if value is None:
            self._property_company = None
            return
        
        self.assert_isinstance(value, "company", six.string_types)
        self._property_company = value

    @schema_property('email')
    def email(self):
        return self._property_email

    @email.setter
    def email(self, value):
        if value is None:
            self._property_email = None
            return
        
        self.assert_isinstance(value, "email", six.string_types)
        self._property_email = value

    @schema_property('provider')
    def provider(self):
        return self._property_provider

    @provider.setter
    def provider(self, value):
        if value is None:
            self._property_provider = None
            return
        
        self.assert_isinstance(value, "provider", six.string_types)
        self._property_provider = value

    @schema_property('provider_user_id')
    def provider_user_id(self):
        return self._property_provider_user_id

    @provider_user_id.setter
    def provider_user_id(self, value):
        if value is None:
            self._property_provider_user_id = None
            return
        
        self.assert_isinstance(value, "provider_user_id", six.string_types)
        self._property_provider_user_id = value

    @schema_property('provider_token')
    def provider_token(self):
        return self._property_provider_token

    @provider_token.setter
    def provider_token(self, value):
        if value is None:
            self._property_provider_token = None
            return
        
        self.assert_isinstance(value, "provider_token", six.string_types)
        self._property_provider_token = value

    @schema_property('given_name')
    def given_name(self):
        return self._property_given_name

    @given_name.setter
    def given_name(self, value):
        if value is None:
            self._property_given_name = None
            return
        
        self.assert_isinstance(value, "given_name", six.string_types)
        self._property_given_name = value

    @schema_property('family_name')
    def family_name(self):
        return self._property_family_name

    @family_name.setter
    def family_name(self, value):
        if value is None:
            self._property_family_name = None
            return
        
        self.assert_isinstance(value, "family_name", six.string_types)
        self._property_family_name = value

    @schema_property('avatar')
    def avatar(self):
        return self._property_avatar

    @avatar.setter
    def avatar(self, value):
        if value is None:
            self._property_avatar = None
            return
        
        self.assert_isinstance(value, "avatar", six.string_types)
        self._property_avatar = value


class AddUserResponse(Response):
    """
    Response of auth.add_user endpoint.

    :param id: New user ID
    :type id: str
    :param secret: The secret key used as the user's password
    :type secret: str
    """
    _service = "auth"
    _action = "add_user"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'id': {'description': 'New user ID', 'type': ['string', 'null']},
            'secret': {
                'description': "The secret key used as the user's password",
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, id=None, secret=None, **kwargs):
        super(AddUserResponse, self).__init__(**kwargs)
        self.id = id
        self.secret = secret

    @schema_property('id')
    def id(self):
        return self._property_id

    @id.setter
    def id(self, value):
        if value is None:
            self._property_id = None
            return
        
        self.assert_isinstance(value, "id", six.string_types)
        self._property_id = value

    @schema_property('secret')
    def secret(self):
        return self._property_secret

    @secret.setter
    def secret(self, value):
        if value is None:
            self._property_secret = None
            return
        
        self.assert_isinstance(value, "secret", six.string_types)
        self._property_secret = value


class CreateCredentialsRequest(Request):
    """
    Creates a new set of credentials for the authenticated user.
                            New key/secret is returned.
                            Note: Secret will never be returned in any other API call.
                            If a secret is lost or compromised, the key should be revoked
                            and a new set of credentials can be created.

    """

    _service = "auth"
    _action = "create_credentials"
    _version = "1.5"
    _schema = {
        'additionalProperties': False,
        'definitions': {},
        'properties': {},
        'type': 'object',
    }


class CreateCredentialsResponse(Response):
    """
    Response of auth.create_credentials endpoint.

    :param credentials: Created credentials
    :type credentials: Credentials
    """
    _service = "auth"
    _action = "create_credentials"
    _version = "1.5"

    _schema = {
        'definitions': {
            'credentials': {
                'properties': {
                    'access_key': {
                        'description': 'Credentials access key',
                        'type': ['string', 'null'],
                    },
                    'secret_key': {
                        'description': 'Credentials secret key',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'credentials': {
                'description': 'Created credentials',
                'oneOf': [{'$ref': '#/definitions/credentials'}, {'type': 'null'}],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, credentials=None, **kwargs):
        super(CreateCredentialsResponse, self).__init__(**kwargs)
        self.credentials = credentials

    @schema_property('credentials')
    def credentials(self):
        return self._property_credentials

    @credentials.setter
    def credentials(self, value):
        if value is None:
            self._property_credentials = None
            return
        if isinstance(value, dict):
            value = Credentials.from_dict(value)
        else:
            self.assert_isinstance(value, "credentials", Credentials)
        self._property_credentials = value


class DeleteUserRequest(Request):
    """
    Delete a new user manually. Only supported in on-premises deployments. This only removes the user's auth entry so that any references to the deleted user's ID will still have valid user information

    :param user: User ID
    :type user: str
    """

    _service = "auth"
    _action = "delete_user"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {'user': {'description': 'User ID', 'type': 'string'}},
        'required': ['user'],
        'type': 'object',
    }
    def __init__(
            self, user, **kwargs):
        super(DeleteUserRequest, self).__init__(**kwargs)
        self.user = user

    @schema_property('user')
    def user(self):
        return self._property_user

    @user.setter
    def user(self, value):
        if value is None:
            self._property_user = None
            return
        
        self.assert_isinstance(value, "user", six.string_types)
        self._property_user = value


class DeleteUserResponse(Response):
    """
    Response of auth.delete_user endpoint.

    :param deleted: True if user was successfully deleted, False otherwise
    :type deleted: bool
    """
    _service = "auth"
    _action = "delete_user"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'deleted': {
                'description': 'True if user was successfully deleted, False otherwise',
                'type': ['boolean', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, deleted=None, **kwargs):
        super(DeleteUserResponse, self).__init__(**kwargs)
        self.deleted = deleted

    @schema_property('deleted')
    def deleted(self):
        return self._property_deleted

    @deleted.setter
    def deleted(self, value):
        if value is None:
            self._property_deleted = None
            return
        
        self.assert_isinstance(value, "deleted", (bool,))
        self._property_deleted = value


class EditUserRequest(Request):
    """
     Edit a users' auth data properties

    :param user: User ID
    :type user: str
    :param role: The new user's role within the company
    :type role: str
    """

    _service = "auth"
    _action = "edit_user"
    _version = "1.9"
    _schema = {
        'definitions': {},
        'properties': {
            'role': {
                'description': "The new user's role within the company",
                'enum': ['admin', 'superuser', 'user', 'annotator'],
                'type': ['string', 'null'],
            },
            'user': {'description': 'User ID', 'type': ['string', 'null']},
        },
        'type': 'object',
    }
    def __init__(
            self, user=None, role=None, **kwargs):
        super(EditUserRequest, self).__init__(**kwargs)
        self.user = user
        self.role = role

    @schema_property('user')
    def user(self):
        return self._property_user

    @user.setter
    def user(self, value):
        if value is None:
            self._property_user = None
            return
        
        self.assert_isinstance(value, "user", six.string_types)
        self._property_user = value

    @schema_property('role')
    def role(self):
        return self._property_role

    @role.setter
    def role(self, value):
        if value is None:
            self._property_role = None
            return
        
        self.assert_isinstance(value, "role", six.string_types)
        self._property_role = value


class EditUserResponse(Response):
    """
    Response of auth.edit_user endpoint.

    :param updated: Number of users updated (0 or 1)
    :type updated: float
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "auth"
    _action = "edit_user"
    _version = "1.9"

    _schema = {
        'definitions': {},
        'properties': {
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'updated': {
                'description': 'Number of users updated (0 or 1)',
                'enum': [0, 1],
                'type': ['number', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, updated=None, fields=None, **kwargs):
        super(EditUserResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property('updated')
    def updated(self):
        return self._property_updated

    @updated.setter
    def updated(self, value):
        if value is None:
            self._property_updated = None
            return
        
        self.assert_isinstance(value, "updated", six.integer_types + (float,))
        self._property_updated = value

    @schema_property('fields')
    def fields(self):
        return self._property_fields

    @fields.setter
    def fields(self, value):
        if value is None:
            self._property_fields = None
            return
        
        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class GetCredentialsRequest(Request):
    """
    Returns all existing credential keys for the authenticated user.
            Note: Only credential keys are returned.

    """

    _service = "auth"
    _action = "get_credentials"
    _version = "1.5"
    _schema = {
        'additionalProperties': False,
        'definitions': {},
        'properties': {},
        'type': 'object',
    }


class GetCredentialsResponse(Response):
    """
    Response of auth.get_credentials endpoint.

    :param credentials: List of credentials, each with an empty secret field.
    :type credentials: Sequence[CredentialKey]
    """
    _service = "auth"
    _action = "get_credentials"
    _version = "1.5"

    _schema = {
        'definitions': {
            'credential_key': {
                'properties': {
                    'access_key': {'description': '', 'type': ['string', 'null']},
                },
                'type': 'object',
            },
        },
        'properties': {
            'credentials': {
                'description': 'List of credentials, each with an empty secret field.',
                'items': {'$ref': '#/definitions/credential_key'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, credentials=None, **kwargs):
        super(GetCredentialsResponse, self).__init__(**kwargs)
        self.credentials = credentials

    @schema_property('credentials')
    def credentials(self):
        return self._property_credentials

    @credentials.setter
    def credentials(self, value):
        if value is None:
            self._property_credentials = None
            return
        
        self.assert_isinstance(value, "credentials", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [CredentialKey.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "credentials", CredentialKey, is_array=True)
        self._property_credentials = value


class GetTaskTokenRequest(Request):
    """
    Get a task-limited token based on supplied credentials (token or key/secret). 
                Intended for use by users who wish to run a task under limited credentials. 
                Returned token will be limited so that all operations can only be performed on the 
            specified task.

    :param task: Task ID
    :type task: str
    :param expiration_sec: Requested token expiration time in seconds. Not
        guaranteed,  might be overridden by the service
    :type expiration_sec: int
    """

    _service = "auth"
    _action = "get_task_token"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'expiration_sec': {
                'description': 'Requested token expiration time in seconds.\n                    Not guaranteed,  might be overridden by the service',
                'type': 'integer',
            },
            'task': {'description': 'Task ID', 'type': 'string'},
        },
        'required': ['task'],
        'type': 'object',
    }
    def __init__(
            self, task, expiration_sec=None, **kwargs):
        super(GetTaskTokenRequest, self).__init__(**kwargs)
        self.task = task
        self.expiration_sec = expiration_sec

    @schema_property('task')
    def task(self):
        return self._property_task

    @task.setter
    def task(self, value):
        if value is None:
            self._property_task = None
            return
        
        self.assert_isinstance(value, "task", six.string_types)
        self._property_task = value

    @schema_property('expiration_sec')
    def expiration_sec(self):
        return self._property_expiration_sec

    @expiration_sec.setter
    def expiration_sec(self, value):
        if value is None:
            self._property_expiration_sec = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "expiration_sec", six.integer_types)
        self._property_expiration_sec = value


class GetTaskTokenResponse(Response):
    """
    Response of auth.get_task_token endpoint.

    :param token: Token string
    :type token: str
    """
    _service = "auth"
    _action = "get_task_token"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'token': {'description': 'Token string', 'type': ['string', 'null']},
        },
        'type': 'object',
    }
    def __init__(
            self, token=None, **kwargs):
        super(GetTaskTokenResponse, self).__init__(**kwargs)
        self.token = token

    @schema_property('token')
    def token(self):
        return self._property_token

    @token.setter
    def token(self, value):
        if value is None:
            self._property_token = None
            return
        
        self.assert_isinstance(value, "token", six.string_types)
        self._property_token = value


class LoginRequest(Request):
    """
    Get a token based on supplied credentials (key/secret).
            Intended for use by users with key/secret credentials that wish to obtain a token
            for use with other services. Token will be limited by the same permissions that
            exist for the credentials used in this call.

    :param expiration_sec: Requested token expiration time in seconds. Not
        guaranteed,  might be overridden by the service
    :type expiration_sec: int
    """

    _service = "auth"
    _action = "login"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'expiration_sec': {
                'description': 'Requested token expiration time in seconds. \n                        Not guaranteed,  might be overridden by the service',
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, expiration_sec=None, **kwargs):
        super(LoginRequest, self).__init__(**kwargs)
        self.expiration_sec = expiration_sec

    @schema_property('expiration_sec')
    def expiration_sec(self):
        return self._property_expiration_sec

    @expiration_sec.setter
    def expiration_sec(self, value):
        if value is None:
            self._property_expiration_sec = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "expiration_sec", six.integer_types)
        self._property_expiration_sec = value


class LoginResponse(Response):
    """
    Response of auth.login endpoint.

    :param token: Token string
    :type token: str
    """
    _service = "auth"
    _action = "login"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'token': {'description': 'Token string', 'type': ['string', 'null']},
        },
        'type': 'object',
    }
    def __init__(
            self, token=None, **kwargs):
        super(LoginResponse, self).__init__(**kwargs)
        self.token = token

    @schema_property('token')
    def token(self):
        return self._property_token

    @token.setter
    def token(self, value):
        if value is None:
            self._property_token = None
            return
        
        self.assert_isinstance(value, "token", six.string_types)
        self._property_token = value


class ReloadConfigRequest(Request):
    """
    Reload auth configuration (currently supports blocking tokens). For user roles associated with a company (Admin, Superuser) this call will only affect company-related configuration.

    """

    _service = "auth"
    _action = "reload_config"
    _version = "1.5"
    _schema = {'definitions': {}, 'properties': {}, 'type': 'object'}


class ReloadConfigResponse(Response):
    """
    Response of auth.reload_config endpoint.

    """
    _service = "auth"
    _action = "reload_config"
    _version = "1.5"

    _schema = {'definitions': {}, 'properties': {}, 'type': 'object'}


class RevokeCredentialsRequest(Request):
    """
    Revokes (and deletes) a set (key, secret) of credentials for
            the authenticated user.

    :param access_key: Credentials key
    :type access_key: str
    """

    _service = "auth"
    _action = "revoke_credentials"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'access_key': {
                'description': 'Credentials key',
                'type': ['string', 'null'],
            },
        },
        'required': ['key_id'],
        'type': 'object',
    }
    def __init__(
            self, access_key=None, **kwargs):
        super(RevokeCredentialsRequest, self).__init__(**kwargs)
        self.access_key = access_key

    @schema_property('access_key')
    def access_key(self):
        return self._property_access_key

    @access_key.setter
    def access_key(self, value):
        if value is None:
            self._property_access_key = None
            return
        
        self.assert_isinstance(value, "access_key", six.string_types)
        self._property_access_key = value


class RevokeCredentialsResponse(Response):
    """
    Response of auth.revoke_credentials endpoint.

    :param revoked: Number of credentials revoked
    :type revoked: int
    """
    _service = "auth"
    _action = "revoke_credentials"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'revoked': {
                'description': 'Number of credentials revoked',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, revoked=None, **kwargs):
        super(RevokeCredentialsResponse, self).__init__(**kwargs)
        self.revoked = revoked

    @schema_property('revoked')
    def revoked(self):
        return self._property_revoked

    @revoked.setter
    def revoked(self, value):
        if value is None:
            self._property_revoked = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "revoked", six.integer_types)
        self._property_revoked = value


class SetCredentialsRequest(Request):
    """
    Set a secret_key for a given access_key. Only supported in on-premises deployments

    :param access_key: Credentials key. Must be identical to the user's ID (this is
        the only value supported in on-premises deployments)
    :type access_key: str
    :param secret_key: New secret key
    :type secret_key: str
    """

    _service = "auth"
    _action = "set_credentials"
    _version = "1.5"
    _schema = {
        'definitions': {},
        'properties': {
            'access_key': {
                'description': "Credentials key. Must be identical to the user's ID (this is the only value supported in on-premises deployments)",
                'type': 'string',
            },
            'secret_key': {'description': 'New secret key', 'type': 'string'},
        },
        'required': ['access_key', 'secret_key'],
        'type': 'object',
    }
    def __init__(
            self, access_key, secret_key, **kwargs):
        super(SetCredentialsRequest, self).__init__(**kwargs)
        self.access_key = access_key
        self.secret_key = secret_key

    @schema_property('access_key')
    def access_key(self):
        return self._property_access_key

    @access_key.setter
    def access_key(self, value):
        if value is None:
            self._property_access_key = None
            return
        
        self.assert_isinstance(value, "access_key", six.string_types)
        self._property_access_key = value

    @schema_property('secret_key')
    def secret_key(self):
        return self._property_secret_key

    @secret_key.setter
    def secret_key(self, value):
        if value is None:
            self._property_secret_key = None
            return
        
        self.assert_isinstance(value, "secret_key", six.string_types)
        self._property_secret_key = value


class SetCredentialsResponse(Response):
    """
    Response of auth.set_credentials endpoint.

    :param set: True if secret was successfully set, False otherwise
    :type set: bool
    """
    _service = "auth"
    _action = "set_credentials"
    _version = "1.5"

    _schema = {
        'definitions': {},
        'properties': {
            'set': {
                'description': 'True if secret was successfully set, False otherwise',
                'type': ['boolean', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, set=None, **kwargs):
        super(SetCredentialsResponse, self).__init__(**kwargs)
        self.set = set

    @schema_property('set')
    def set(self):
        return self._property_set

    @set.setter
    def set(self, value):
        if value is None:
            self._property_set = None
            return
        
        self.assert_isinstance(value, "set", (bool,))
        self._property_set = value


response_mapping = {
    LoginRequest: LoginResponse,
    GetTaskTokenRequest: GetTaskTokenResponse,
    CreateCredentialsRequest: CreateCredentialsResponse,
    GetCredentialsRequest: GetCredentialsResponse,
    RevokeCredentialsRequest: RevokeCredentialsResponse,
    SetCredentialsRequest: SetCredentialsResponse,
    AddUserRequest: AddUserResponse,
    DeleteUserRequest: DeleteUserResponse,
    ReloadConfigRequest: ReloadConfigResponse,
    EditUserRequest: EditUserResponse,
}
