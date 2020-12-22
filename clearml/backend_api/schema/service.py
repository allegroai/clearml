import itertools
import re

import attr
import six

from ...utilities.pyhocon import ConfigTree
from .action import Action


class Service(object):
    """ Service schema handler """

    __jsonschema_ref_ex = re.compile("^#/definitions/(.*)$")

    @property
    def default(self):
        return self._default

    @property
    def actions(self):
        return self._actions

    @property
    def definitions(self):
        """ Raw  service definitions (each might be dependant on some of its siblings) """
        return self._definitions

    @property
    def definitions_refs(self):
        return self._definitions_refs

    @property
    def name(self):
        return self._name

    @property
    def doc(self):
        return self._doc

    def __init__(self, name, service_config):
        self._name = name
        self._default = None
        self._actions = []
        self._definitions = None
        self._definitions_refs = None
        self._doc = None
        self.parse(service_config)

    @classmethod
    def get_ref_name(cls, ref_string):
        m = cls.__jsonschema_ref_ex.match(ref_string)
        if m:
            return m.group(1)

    def parse(self, service_config):
        self._default = service_config.get(
            "_default", ConfigTree()
        ).as_plain_ordered_dict()

        self._doc = '{} service'.format(self.name)
        description = service_config.get('_description', '')
        if description:
            self._doc += '\n\n{}'.format(description)
        self._definitions = service_config.get(
            "_definitions", ConfigTree()
        ).as_plain_ordered_dict()
        self._definitions_refs = {
            k: self._get_schema_references(v) for k, v in self._definitions.items()
        }
        all_refs = set(itertools.chain(*self.definitions_refs.values()))
        if not all_refs.issubset(self.definitions):
            raise ValueError(
                "Unresolved references (%s) in %s/definitions"
                % (", ".join(all_refs.difference(self.definitions)), self.name)
            )

        actions = {
            k: v.as_plain_ordered_dict()
            for k, v in service_config.items()
            if not k.startswith("_")
        }
        self._actions = {
            action_name: action
            for action_name, action in (
                (action_name, self._parse_action_versions(action_name, action_versions))
                for action_name, action_versions in actions.items()
            )
            if action
        }

    def _parse_action_versions(self, action_name, action_versions):
        def parse_version(action_version):
            try:
                return float(action_version)
            except (ValueError, TypeError) as ex:
                raise ValueError(
                    "Failed parsing version number {} ({}) in {}/{}".format(
                        action_version, ex.args[0], self.name, action_name
                    )
                )

        def add_internal(cfg):
            if "internal" in action_versions:
                cfg.setdefault("internal", action_versions["internal"])
            return cfg

        return {
            parsed_version: action
            for parsed_version, action in (
                (parsed_version, self._parse_action(action_name, parsed_version, add_internal(cfg)))
                for parsed_version, cfg in (
                    (parse_version(version), cfg)
                    for version, cfg in action_versions.items()
                    if version not in ["internal", "allow_roles", "authorize"]
                )
            )
            if action
        }

    def _get_schema_references(self, s):
        refs = set()
        if isinstance(s, dict):
            for k, v in s.items():
                if isinstance(v, six.string_types):
                    m = self.__jsonschema_ref_ex.match(v)
                    if m:
                        refs.add(m.group(1))
                    continue
                elif k in ("oneOf", "anyOf") and isinstance(v, list):
                    refs.update(*map(self._get_schema_references, v))
                refs.update(self._get_schema_references(v))
        return refs

    def _expand_schema_references_with_definitions(self, schema, refs=None):
        definitions = schema.get("definitions", {})
        refs = refs if refs is not None else self._get_schema_references(schema)
        required_refs = set(refs).difference(definitions)
        if not required_refs:
            return required_refs
        if not required_refs.issubset(self.definitions):
            raise ValueError(
                "Unresolved references (%s)"
                % ", ".join(required_refs.difference(self.definitions))
            )

        # update required refs with all sub requirements
        last_required_refs = None
        while last_required_refs != required_refs:
            last_required_refs = required_refs.copy()
            additional_refs = set(
                itertools.chain(
                    *(self.definitions_refs.get(ref, []) for ref in required_refs)
                )
            )
            required_refs.update(additional_refs)
        return required_refs

    def _resolve_schema_references(self, schema, refs=None):
        definitions = schema.get("definitions", {})
        definitions.update({k: v for k, v in self.definitions.items() if k in refs})
        schema["definitions"] = definitions

    def _parse_action(self, action_name, action_version, action_config):
        data = self.default.copy()
        data.update(action_config)

        if not action_config.get("generate", True):
            return None

        definitions_keys = set()
        for schema_key in ("request", "response"):
            if schema_key in action_config:
                try:
                    schema = action_config[schema_key]
                    refs = self._expand_schema_references_with_definitions(schema)
                    self._resolve_schema_references(schema, refs=refs)
                    definitions_keys.update(refs)
                except ValueError as ex:
                    name = "%s.%s/%.1f/%s" % (
                        self.name,
                        action_name,
                        action_version,
                        schema_key,
                    )
                    raise ValueError("%s in %s" % (str(ex), name))

        return Action(
            name=action_name,
            version=action_version,
            definitions_keys=list(definitions_keys),
            service=self.name,
            **(
                {
                    key: value
                    for key, value in data.items()
                    if key in attr.fields_dict(Action)
                }
            )
        )
