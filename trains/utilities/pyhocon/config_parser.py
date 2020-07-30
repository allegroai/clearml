import itertools
import re
import os
import socket
import contextlib
import codecs
from datetime import timedelta

from pyparsing import Forward, Keyword, QuotedString, Word, Literal, Suppress, Regex, Optional, SkipTo, ZeroOrMore, \
    Group, lineno, col, TokenConverter, replaceWith, alphanums, alphas8bit, ParseSyntaxException, StringEnd
from pyparsing import ParserElement
from .config_tree import ConfigTree, ConfigSubstitution, ConfigList, ConfigValues, ConfigUnquotedString, \
    ConfigInclude, NoneValue, ConfigQuotedString
from .exceptions import ConfigSubstitutionException, ConfigMissingException, ConfigException
import logging
import copy

use_urllib2 = False
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
    from urllib.error import HTTPError, URLError
except ImportError:  # pragma: no cover
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen, HTTPError, URLError

    use_urllib2 = True
try:
    basestring
except NameError:  # pragma: no cover
    basestring = str
    unicode = str

logger = logging.getLogger(__name__)

#
# Substitution Defaults
#


class DEFAULT_SUBSTITUTION(object):
    pass


class MANDATORY_SUBSTITUTION(object):
    pass


class NO_SUBSTITUTION(object):
    pass


class STR_SUBSTITUTION(object):
    pass


def period(period_value, period_unit):
    try:
        from dateutil.relativedelta import relativedelta as period_impl
    except Exception:
        from datetime import timedelta as period_impl

    if period_unit == 'nanoseconds':
        period_unit = 'microseconds'
        period_value = int(period_value / 1000)

    arguments = dict(zip((period_unit,), (period_value,)))

    if period_unit == 'milliseconds':
        return timedelta(**arguments)

    return period_impl(**arguments)


class ConfigFactory(object):

    @classmethod
    def parse_file(cls, filename, encoding='utf-8', required=True, resolve=True, unresolved_value=DEFAULT_SUBSTITUTION):
        """Parse file

        :param filename: filename
        :type filename: basestring
        :param encoding: file encoding
        :type encoding: basestring
        :param required: If true, raises an exception if can't load file
        :type required: boolean
        :param resolve: if true, resolve substitutions
        :type resolve: boolean
        :param unresolved_value: assigned value value to unresolved substitution.
            If overriden with a default value, it will replace all unresolved value to the default value.
            If it is set to to pyhocon.STR_SUBSTITUTION then it will replace the value by its
            substitution expression (e.g., ${x})
        :type unresolved_value: class
        :return: Config object
        :type return: Config
        """
        try:
            with codecs.open(filename, 'r', encoding=encoding) as fd:
                content = fd.read()
                return cls.parse_string(content, os.path.dirname(filename), resolve, unresolved_value)
        except IOError as e:
            if required:
                raise e
            logger.warn('Cannot include file %s. File does not exist or cannot be read.', filename)
            return []

    @classmethod
    def parse_URL(cls, url, timeout=None, resolve=True, required=False, unresolved_value=DEFAULT_SUBSTITUTION):
        """Parse URL

        :param url: url to parse
        :type url: basestring
        :param resolve: if true, resolve substitutions
        :type resolve: boolean
        :param unresolved_value: assigned value value to unresolved substitution.
            If overriden with a default value, it will replace all unresolved value to the default value.
            If it is set to to pyhocon.STR_SUBSTITUTION then it will replace the value by
            its substitution expression (e.g., ${x})
        :type unresolved_value: boolean
        :return: Config object or []
        :type return: Config or list
        """
        socket_timeout = socket._GLOBAL_DEFAULT_TIMEOUT if timeout is None else timeout

        try:
            with contextlib.closing(urlopen(url, timeout=socket_timeout)) as fd:
                content = fd.read() if use_urllib2 else fd.read().decode('utf-8')
                return cls.parse_string(content, os.path.dirname(url), resolve, unresolved_value)
        except (HTTPError, URLError) as e:
            logger.warn('Cannot include url %s. Resource is inaccessible.', url)
            if required:
                raise e
            else:
                return []

    @classmethod
    def parse_string(cls, content, basedir=None, resolve=True, unresolved_value=DEFAULT_SUBSTITUTION):
        """Parse URL

        :param content: content to parse
        :type content: basestring
        :param resolve: If true, resolve substitutions
        :param resolve: if true, resolve substitutions
        :type resolve: boolean
        :param unresolved_value: assigned value value to unresolved substitution.
            If overriden with a default value, it will replace all unresolved value to the default value.
            If it is set to to pyhocon.STR_SUBSTITUTION then it will replace the value by
            its substitution expression (e.g., ${x})
        :type unresolved_value: boolean
        :return: Config object
        :type return: Config
        """
        return ConfigParser().parse(content, basedir, resolve, unresolved_value)

    @classmethod
    def from_dict(cls, dictionary, root=False):
        """Convert dictionary (and ordered dictionary) into a ConfigTree
        :param dictionary: dictionary to convert
        :type dictionary: dict
        :return: Config object
        :type return: Config
        """

        def create_tree(value):
            if isinstance(value, dict):
                res = ConfigTree(root=root)
                for key, child_value in value.items():
                    res.put(key, create_tree(child_value))
                return res
            if isinstance(value, list):
                return [create_tree(v) for v in value]
            else:
                return value

        return create_tree(dictionary)


class ConfigParser(object):
    """
    Parse HOCON files: https://github.com/typesafehub/config/blob/master/HOCON.md
    """

    REPLACEMENTS = {
        '\\\\': '\\',
        '\\\n': '\n',
        '\\n': '\n',
        '\\r': '\r',
        '\\t': '\t',
        '\\=': '=',
        '\\#': '#',
        '\\!': '!',
        '\\"': '"',
    }

    period_type_map = {
        'nanoseconds': ['ns', 'nano', 'nanos', 'nanosecond', 'nanoseconds'],

        'microseconds': ['us', 'micro', 'micros', 'microsecond', 'microseconds'],
        'milliseconds': ['ms', 'milli', 'millis', 'millisecond', 'milliseconds'],
        'seconds': ['s', 'second', 'seconds'],
        'minutes': ['m', 'minute', 'minutes'],
        'hours': ['h', 'hour', 'hours'],
        'weeks': ['w', 'week', 'weeks'],
        'days': ['d', 'day', 'days'],

    }

    optional_period_type_map = {
        'months': ['mo', 'month', 'months'],  # 'm' from hocon spec removed. conflicts with minutes syntax.
        'years': ['y', 'year', 'years']
    }

    supported_period_map = None

    @classmethod
    def get_supported_period_type_map(cls):
        if cls.supported_period_map is None:
            cls.supported_period_map = {}
            cls.supported_period_map.update(cls.period_type_map)

            try:
                from dateutil import relativedelta

                if relativedelta is not None:
                    cls.supported_period_map.update(cls.optional_period_type_map)
            except Exception:
                pass

        return cls.supported_period_map

    @classmethod
    def parse(cls, content, basedir=None, resolve=True, unresolved_value=DEFAULT_SUBSTITUTION):
        """parse a HOCON content

        :param content: HOCON content to parse
        :type content: basestring
        :param resolve: if true, resolve substitutions
        :type resolve: boolean
        :param unresolved_value: assigned value value to unresolved substitution.
            If overriden with a default value, it will replace all unresolved value to the default value.
            If it is set to to pyhocon.STR_SUBSTITUTION then it will replace the value by
            its substitution expression (e.g., ${x})
        :type unresolved_value: boolean
        :return: a ConfigTree or a list
        """

        unescape_pattern = re.compile(r'\\.')

        def replace_escape_sequence(match):
            value = match.group(0)
            return cls.REPLACEMENTS.get(value, value)

        def norm_string(value):
            return unescape_pattern.sub(replace_escape_sequence, value)

        def unescape_string(tokens):
            return ConfigUnquotedString(norm_string(tokens[0]))

        def parse_multi_string(tokens):
            # remove the first and last 3 "
            return tokens[0][3: -3]

        def convert_number(tokens):
            n = tokens[0]
            try:
                return int(n, 10)
            except ValueError:
                return float(n)

        def safe_convert_number(tokens):
            n = tokens[0]
            try:
                return int(n, 10)
            except ValueError:
                try:
                    return float(n)
                except ValueError:
                    return n

        def convert_period(tokens):

            period_value = int(tokens.value)
            period_identifier = tokens.unit

            period_unit = next((single_unit for single_unit, values
                                in cls.get_supported_period_type_map().items()
                                if period_identifier in values))

            return period(period_value, period_unit)

        # ${path} or ${?path} for optional substitution
        SUBSTITUTION_PATTERN = r"\$\{(?P<optional>\?)?(?P<variable>[^}]+)\}(?P<ws>[ \t]*)"

        def create_substitution(instring, loc, token):
            # remove the ${ and }
            match = re.match(SUBSTITUTION_PATTERN, token[0])
            variable = match.group('variable')
            ws = match.group('ws')
            optional = match.group('optional') == '?'
            substitution = ConfigSubstitution(variable, optional, ws, instring, loc)
            return substitution

        # ${path} or ${?path} for optional substitution
        STRING_PATTERN = '"(?P<value>(?:[^"\\\\]|\\\\.)*)"(?P<ws>[ \t]*)'

        def create_quoted_string(instring, loc, token):
            # remove the ${ and }
            match = re.match(STRING_PATTERN, token[0])
            value = norm_string(match.group('value'))
            ws = match.group('ws')
            return ConfigQuotedString(value, ws, instring, loc)

        def include_config(instring, loc, token):
            url = None
            file = None
            required = False

            if token[0] == 'required':
                required = True
                final_tokens = token[1:]
            else:
                final_tokens = token

            if len(final_tokens) == 1:  # include "test"
                value = final_tokens[0].value if isinstance(final_tokens[0], ConfigQuotedString) else final_tokens[0]
                if value.startswith("http://") or value.startswith("https://") or value.startswith("file://"):
                    url = value
                else:
                    file = value
            elif len(final_tokens) == 2:  # include url("test") or file("test")
                value = final_tokens[1].value if isinstance(token[1], ConfigQuotedString) else final_tokens[1]
                if final_tokens[0] == 'url':
                    url = value
                else:
                    file = value

            if url is not None:
                logger.debug('Loading config from url %s', url)
                obj = ConfigFactory.parse_URL(
                    url,
                    resolve=False,
                    required=required,
                    unresolved_value=NO_SUBSTITUTION
                )
            elif file is not None:
                path = file if basedir is None else os.path.join(basedir, file)
                logger.debug('Loading config from file %s', path)
                obj = ConfigFactory.parse_file(
                    path,
                    resolve=False,
                    required=required,
                    unresolved_value=NO_SUBSTITUTION
                )
            else:
                raise ConfigException('No file or URL specified at: {loc}: {instring}', loc=loc, instring=instring)

            return ConfigInclude(obj if isinstance(obj, list) else obj.items())

        @contextlib.contextmanager
        def set_default_white_spaces():
            default = ParserElement.DEFAULT_WHITE_CHARS
            ParserElement.setDefaultWhitespaceChars(' \t')
            yield
            ParserElement.setDefaultWhitespaceChars(default)

        with set_default_white_spaces():
            assign_expr = Forward()
            true_expr = Keyword("true", caseless=True).setParseAction(replaceWith(True))
            false_expr = Keyword("false", caseless=True).setParseAction(replaceWith(False))
            null_expr = Keyword("null", caseless=True).setParseAction(replaceWith(NoneValue()))
            # key = QuotedString('"', escChar='\\', unquoteResults=False) | Word(alphanums + alphas8bit + '._- /')
            regexp_numbers = r'[+-]?(\d*\.\d+|\d+(\.\d+)?)([eE][+\-]?\d+)?(?=$|[ \t]*([\$\}\],#\n\r]|//))'
            key = QuotedString('"', escChar='\\', unquoteResults=False) | \
                Regex(regexp_numbers, re.DOTALL).setParseAction(safe_convert_number) | \
                Word(alphanums + alphas8bit + '._- /')

            eol = Word('\n\r').suppress()
            eol_comma = Word('\n\r,').suppress()
            comment = (Literal('#') | Literal('//')) - SkipTo(eol | StringEnd())
            comment_eol = Suppress(Optional(eol_comma) + comment)
            comment_no_comma_eol = (comment | eol).suppress()
            number_expr = Regex(regexp_numbers, re.DOTALL).setParseAction(convert_number)

            period_types = itertools.chain.from_iterable(cls.get_supported_period_type_map().values())
            period_expr = Regex(r'(?P<value>\d+)\s*(?P<unit>' + '|'.join(period_types) + ')$'
                                ).setParseAction(convert_period)

            # multi line string using """
            # Using fix described in http://pyparsing.wikispaces.com/share/view/3778969
            multiline_string = Regex('""".*?"*"""', re.DOTALL | re.UNICODE).setParseAction(parse_multi_string)
            # single quoted line string
            quoted_string = Regex(r'"(?:[^"\\\n]|\\.)*"[ \t]*', re.UNICODE).setParseAction(create_quoted_string)
            # unquoted string that takes the rest of the line until an optional comment
            # we support .properties multiline support which is like this:
            # line1  \
            # line2 \
            # so a backslash precedes the \n
            unquoted_string = Regex(r'(?:[^^`+?!@*&"\[\{\s\]\}#,=\$\\]|\\.)+[ \t]*',
                                    re.UNICODE).setParseAction(unescape_string)
            substitution_expr = Regex(r'[ \t]*\$\{[^\}]+\}[ \t]*').setParseAction(create_substitution)
            string_expr = multiline_string | quoted_string | unquoted_string

            value_expr = period_expr | number_expr | true_expr | false_expr | null_expr | string_expr

            include_content = (quoted_string | ((Keyword('url') | Keyword(
                'file')) - Literal('(').suppress() - quoted_string - Literal(')').suppress()))
            include_expr = (
                Keyword("include", caseless=True).suppress() + (
                    include_content | (
                        Keyword("required") - Literal('(').suppress() - include_content - Literal(')').suppress()
                    )
                )
            ).setParseAction(include_config)

            root_dict_expr = Forward()
            dict_expr = Forward()
            list_expr = Forward()
            multi_value_expr = ZeroOrMore(comment_eol | include_expr | substitution_expr |
                                          dict_expr | list_expr | value_expr | (Literal('\\') - eol).suppress())
            # for a dictionary : or = is optional
            # last zeroOrMore is because we can have t = {a:4} {b: 6} {c: 7} which is dictionary concatenation
            inside_dict_expr = ConfigTreeParser(ZeroOrMore(comment_eol | include_expr | assign_expr | eol_comma))
            inside_root_dict_expr = ConfigTreeParser(ZeroOrMore(
                comment_eol | include_expr | assign_expr | eol_comma), root=True)
            dict_expr << Suppress('{') - inside_dict_expr - Suppress('}')
            root_dict_expr << Suppress('{') - inside_root_dict_expr - Suppress('}')
            list_entry = ConcatenatedValueParser(multi_value_expr)
            list_expr << Suppress('[') - ListParser(list_entry - ZeroOrMore(eol_comma - list_entry)) - Suppress(']')

            # special case when we have a value assignment where the string can potentially be the remainder of the line
            assign_expr << Group(key - ZeroOrMore(comment_no_comma_eol) -
                                 (dict_expr | (Literal('=') | Literal(':') | Literal('+=')) -
                                  ZeroOrMore(comment_no_comma_eol) - ConcatenatedValueParser(multi_value_expr)))

            # the file can be { ... } where {} can be omitted or []
            config_expr = ZeroOrMore(comment_eol | eol) + (list_expr | root_dict_expr |
                                                           inside_root_dict_expr) + ZeroOrMore(comment_eol | eol_comma)
            config = config_expr.parseString(content, parseAll=True)[0]

            if resolve:
                allow_unresolved = resolve and unresolved_value is not DEFAULT_SUBSTITUTION and \
                                   unresolved_value is not MANDATORY_SUBSTITUTION
                has_unresolved = cls.resolve_substitutions(config, allow_unresolved)
                if has_unresolved and unresolved_value is MANDATORY_SUBSTITUTION:
                    raise ConfigSubstitutionException(
                        'resolve cannot be set to True and unresolved_value to MANDATORY_SUBSTITUTION')

            if unresolved_value is not NO_SUBSTITUTION and unresolved_value is not DEFAULT_SUBSTITUTION:
                cls.unresolve_substitutions_to_value(config, unresolved_value)
        return config

    @classmethod
    def _resolve_variable(cls, config, substitution):
        """
        :param config:
        :param substitution:
        :return: (is_resolved, resolved_variable)
        """
        variable = substitution.variable
        try:
            return True, config.get(variable)
        except ConfigMissingException:
            # default to environment variable
            value = os.environ.get(variable)

            if value is None:
                if substitution.optional:
                    return False, None
                else:
                    raise ConfigSubstitutionException(
                        "Cannot resolve variable ${{{variable}}} (line: {line}, col: {col})".format(
                            variable=variable,
                            line=lineno(substitution.loc, substitution.instring),
                            col=col(substitution.loc, substitution.instring)))
            elif isinstance(value, ConfigList) or isinstance(value, ConfigTree):
                raise ConfigSubstitutionException(
                    "Cannot substitute variable ${{{variable}}} because it does not point to a "
                    "string, int, float, boolean or null {type} (line:{line}, col: {col})".format(
                        variable=variable,
                        type=value.__class__.__name__,
                        line=lineno(substitution.loc, substitution.instring),
                        col=col(substitution.loc, substitution.instring)))
            return True, value

    @classmethod
    def _fixup_self_references(cls, config, accept_unresolved=False):
        if isinstance(config, ConfigTree) and config.root:
            for key in config:  # Traverse history of element
                history = config.history[key]
                previous_item = history[0]
                for current_item in history[1:]:
                    for substitution in cls._find_substitutions(current_item):
                        prop_path = ConfigTree.parse_key(substitution.variable)
                        if len(prop_path) > 1 and config.get(substitution.variable, None) is not None:
                            continue  # If value is present in latest version, don't do anything
                        if prop_path[0] == key:
                            if isinstance(previous_item, ConfigValues) and not accept_unresolved:
                                # We hit a dead end, we cannot evaluate
                                raise ConfigSubstitutionException(
                                    "Property {variable} cannot be substituted. Check for cycles.".format(
                                        variable=substitution.variable
                                    )
                                )
                            else:
                                value = previous_item if len(
                                    prop_path) == 1 else previous_item.get(".".join(prop_path[1:]))
                                _, _, current_item = cls._do_substitute(substitution, value)
                    previous_item = current_item

                if len(history) == 1:
                    for substitution in cls._find_substitutions(previous_item):
                        prop_path = ConfigTree.parse_key(substitution.variable)
                        if len(prop_path) > 1 and config.get(substitution.variable, None) is not None:
                            continue  # If value is present in latest version, don't do anything
                        if prop_path[0] == key and substitution.optional:
                            cls._do_substitute(substitution, None)
                        if prop_path[0] == key:
                            value = os.environ.get(key)
                            if value is not None:
                                cls._do_substitute(substitution, value)
                                continue
                            if substitution.optional:  # special case, when self optional referencing without existing
                                cls._do_substitute(substitution, None)

    # traverse config to find all the substitutions
    @classmethod
    def _find_substitutions(cls, item):
        """Convert HOCON input into a JSON output

        :return: JSON string representation
        :type return: basestring
        """
        if isinstance(item, ConfigValues):
            return item.get_substitutions()

        substitutions = []
        elements = []
        if isinstance(item, ConfigTree):
            elements = item.values()
        elif isinstance(item, list):
            elements = item

        for child in elements:
            substitutions += cls._find_substitutions(child)
        return substitutions

    @classmethod
    def _do_substitute(cls, substitution, resolved_value, is_optional_resolved=True):
        unresolved = False
        new_substitutions = []
        if isinstance(resolved_value, ConfigValues):
            resolved_value = resolved_value.transform()
        if isinstance(resolved_value, ConfigValues):
            unresolved = True
            result = resolved_value
        else:
            # replace token by substitution
            config_values = substitution.parent
            # if it is a string, then add the extra ws that was present in the original string after the substitution
            formatted_resolved_value = resolved_value \
                if resolved_value is None \
                or isinstance(resolved_value, (dict, list)) \
                or substitution.index == len(config_values.tokens) - 1 \
                else (str(resolved_value) + substitution.ws)
            # use a deepcopy of resolved_value to avoid mutation
            config_values.put(substitution.index, copy.deepcopy(formatted_resolved_value))
            transformation = config_values.transform()
            result = config_values.overriden_value \
                if transformation is None and not is_optional_resolved \
                else transformation

            if result is None and config_values.key in config_values.parent:
                del config_values.parent[config_values.key]
            else:
                config_values.parent[config_values.key] = result
                s = cls._find_substitutions(result)
                if s:
                    new_substitutions = s
                    unresolved = True

        return (unresolved, new_substitutions, result)

    @classmethod
    def _final_fixup(cls, item):
        if isinstance(item, ConfigValues):
            return item.transform()
        elif isinstance(item, list):
            return list([cls._final_fixup(child) for child in item])
        elif isinstance(item, ConfigTree):
            items = list(item.items())
            for key, child in items:
                item[key] = cls._final_fixup(child)
        return item

    @classmethod
    def unresolve_substitutions_to_value(cls, config, unresolved_value=STR_SUBSTITUTION):
        for substitution in cls._find_substitutions(config):
            if unresolved_value is STR_SUBSTITUTION:
                value = substitution.raw_str()
            elif unresolved_value is None:
                value = NoneValue()
            else:
                value = unresolved_value
            cls._do_substitute(substitution, value, False)
        cls._final_fixup(config)

    @classmethod
    def resolve_substitutions(cls, config, accept_unresolved=False):
        has_unresolved = False
        cls._fixup_self_references(config, accept_unresolved)
        substitutions = cls._find_substitutions(config)
        if len(substitutions) > 0:
            unresolved = True
            any_unresolved = True
            _substitutions = []
            cache = {}
            while any_unresolved and len(substitutions) > 0 and set(substitutions) != set(_substitutions):
                unresolved = False
                any_unresolved = True
                _substitutions = substitutions[:]

                for substitution in _substitutions:
                    is_optional_resolved, resolved_value = cls._resolve_variable(config, substitution)

                    # if the substitution is optional
                    if not is_optional_resolved and substitution.optional:
                        resolved_value = None
                    if isinstance(resolved_value, ConfigValues):
                        parents = cache.get(resolved_value)
                        if parents is None:
                            parents = []
                            link = resolved_value
                            while isinstance(link, ConfigValues):
                                parents.append(link)
                                link = link.overriden_value
                            cache[resolved_value] = parents

                    if isinstance(resolved_value, ConfigValues) \
                       and substitution.parent in parents \
                       and hasattr(substitution.parent, 'overriden_value') \
                       and substitution.parent.overriden_value:

                        # self resolution, backtrack
                        resolved_value = substitution.parent.overriden_value

                    unresolved, new_substitutions, result = cls._do_substitute(
                        substitution, resolved_value, is_optional_resolved)
                    any_unresolved = unresolved or any_unresolved
                    substitutions.extend(new_substitutions)
                    if not isinstance(result, ConfigValues):
                        substitutions.remove(substitution)

            cls._final_fixup(config)
            if unresolved:
                has_unresolved = True
                if not accept_unresolved:
                    raise ConfigSubstitutionException("Cannot resolve {variables}. Check for cycles.".format(
                        variables=', '.join('${{{variable}}}: (line: {line}, col: {col})'.format(
                            variable=substitution.variable,
                            line=lineno(substitution.loc, substitution.instring),
                            col=col(substitution.loc, substitution.instring)) for substitution in substitutions)))

        cls._final_fixup(config)
        return has_unresolved


class ListParser(TokenConverter):
    """Parse a list [elt1, etl2, ...]
    """

    def __init__(self, expr=None):
        super(ListParser, self).__init__(expr)
        self.saveAsList = True

    def postParse(self, instring, loc, token_list):
        """Create a list from the tokens

        :param instring:
        :param loc:
        :param token_list:
        :return:
        """
        cleaned_token_list = [token for tokens in (token.tokens if isinstance(token, ConfigInclude) else [token]
                                                   for token in token_list if token != '')
                              for token in tokens]
        config_list = ConfigList(cleaned_token_list)
        return [config_list]


class ConcatenatedValueParser(TokenConverter):
    def __init__(self, expr=None):
        super(ConcatenatedValueParser, self).__init__(expr)
        self.parent = None
        self.key = None

    def postParse(self, instring, loc, token_list):
        config_values = ConfigValues(token_list, instring, loc)
        return [config_values.transform()]


class ConfigTreeParser(TokenConverter):
    """
    Parse a config tree from tokens
    """

    def __init__(self, expr=None, root=False):
        super(ConfigTreeParser, self).__init__(expr)
        self.root = root
        self.saveAsList = True

    def postParse(self, instring, loc, token_list):
        """Create ConfigTree from tokens

        :param instring:
        :param loc:
        :param token_list:
        :return:
        """
        config_tree = ConfigTree(root=self.root)
        for element in token_list:
            expanded_tokens = element.tokens if isinstance(element, ConfigInclude) else [element]

            for tokens in expanded_tokens:
                # key, value1 (optional), ...
                key = tokens[0].strip() if isinstance(tokens[0], (unicode, basestring)) else tokens[0]
                operator = '='
                if len(tokens) == 3 and tokens[1].strip() in [':', '=', '+=']:
                    operator = tokens[1].strip()
                    values = tokens[2:]
                elif len(tokens) == 2:
                    values = tokens[1:]
                else:
                    raise ParseSyntaxException("Unknown tokens {tokens} received".format(tokens=tokens))
                # empty string
                if len(values) == 0:
                    config_tree.put(key, '')
                else:
                    value = values[0]
                    if isinstance(value, list) and operator == "+=":
                        value = ConfigValues([ConfigSubstitution(key, True, '', False, loc), value], False, loc)
                        config_tree.put(key, value, False)
                    elif isinstance(value, unicode) and operator == "+=":
                        value = ConfigValues([ConfigSubstitution(key, True, '', True, loc), ' ' + value], True, loc)
                        config_tree.put(key, value, False)
                    elif isinstance(value, list):
                        config_tree.put(key, value, False)
                    else:
                        existing_value = config_tree.get(key, None)
                        if isinstance(value, ConfigTree) and not isinstance(existing_value, list):
                            # Only Tree has to be merged with tree
                            config_tree.put(key, value, True)
                        elif isinstance(value, ConfigValues):
                            conf_value = value
                            value.parent = config_tree
                            value.key = key
                            if isinstance(existing_value, list) or isinstance(existing_value, ConfigTree):
                                config_tree.put(key, conf_value, True)
                            else:
                                config_tree.put(key, conf_value, False)
                        else:
                            config_tree.put(key, value, False)
        return config_tree
