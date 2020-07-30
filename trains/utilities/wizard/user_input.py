from typing import Optional


def get_input(
    key,  # type: str
    description="",  # type: str
    question="Enter",  # type: str
    required=False,  # type: bool
    default=None,  # type: Optional[str]
    new_line=False,  # type: bool
):
    # type: (...) -> Optional[str]
    if new_line:
        print()
    while True:
        value = input("{} {} {}: ".format(question, key, description))
        if not value.strip() and required:
            print("{} is required".format(key))
        elif not (value.strip() or required):
            return default
        else:
            return value


def input_int(
    key,  # type: str
    description="",  # type: str
    required=False,  # type: bool
    default=None,  # type: Optional[int]
    new_line=False,  # type: bool
):
    # type: (...) -> Optional[int]
    while True:
        try:
            value = int(
                get_input(
                    key,
                    description,
                    required=required,
                    default=default,
                    new_line=new_line,
                )
            )
            return value
        except ValueError:
            print(
                "Invalid input: {} should be a number. Please enter an integer".format(
                    key
                )
            )


def input_bool(question, default=False):
    # type: (str, bool) -> bool
    """
    :param question: string to display
    :param default: default boolean value
    :return: return True if response is 'y'/'yes' 't'/'true' in input.lower()
    """
    while True:
        try:
            response = input("{}: ".format(question)).lower()
            if not response:
                return default
            if response.startswith("y") or response.startswith("t"):
                return True
            if response.startswith("n") or response.startswith("f"):
                return False
            raise ValueError()
        except ValueError:
            print("Invalid input: please enter 'yes' or 'no'")
